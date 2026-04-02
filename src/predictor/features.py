"""特徴量エンジニアリング v2

Codex/Gemini の提案を統合。18特徴量 → 約40特徴量に拡張。
主な強化:
  1. ライン戦力分析（ユニットとして評価）
  2. 直近走成績トレンド（指数平滑・回帰傾き）
  3. バンク×脚質の相性分析
  4. レース内相対評価（z-score, rank）
"""
import numpy as np
import pandas as pd
from sqlalchemy import and_
from sqlalchemy.orm import Session

from src.common.database import Odds, Player, Race, RaceEntry, Racecourse


def build_features_for_race(session: Session, race: Race) -> pd.DataFrame:
    """レースの全エントリーから特徴量DataFrameを生成"""
    entries = session.query(RaceEntry).filter_by(race_id=race.id).all()
    racecourse = session.query(Racecourse).filter_by(id=race.racecourse_id).first()

    if not entries:
        return pd.DataFrame()

    # ライン情報を先に構築
    line_info = _build_line_info(entries)

    # レース全体のペース特徴量
    pace_features = _calc_pace_features(entries)

    rows = []
    for entry in entries:
        player = session.query(Player).filter_by(id=entry.player_id).first() if entry.player_id else None
        history = _get_player_history(session, entry.player_id, race.id, limit=10) if entry.player_id else []
        row = _build_entry_features(entry, player, race, racecourse, entries, line_info, history, session)
        row.update(pace_features)
        rows.append(row)

    df = pd.DataFrame(rows)
    # レース内相対特徴量を追加
    df = _add_relative_features(df)
    return df


# =============================================================
# ライン分析
# =============================================================

def _build_line_info(entries: list[RaceEntry]) -> dict:
    """ラインのグループ情報を構築

    Returns:
        {car_number: {"line_id": int, "line_pos": int, "line_size": int,
                       "line_members": [RaceEntry], "is_solo": bool,
                       "line_leader_strength": float, ...}}
    """
    # line_label でグループ化（空なら脚質で推定）
    groups: dict[str, list[RaceEntry]] = {}
    for e in entries:
        label = (e.line_label or "").strip()
        if not label:
            # ラインラベルが無い場合は単騎として扱う
            label = f"_solo_{e.car_number}"
        groups.setdefault(label, []).append(e)

    info = {}
    for line_id, (label, members) in enumerate(groups.items()):
        # ライン内を脚質で並べる（先行→番手→3番手）
        members_sorted = sorted(members, key=lambda e: _style_order(e.line_group))

        line_strengths = [_calc_strength(e) for e in members_sorted]
        line_avg = np.mean(line_strengths) if line_strengths else 0
        line_top = max(line_strengths) if line_strengths else 0
        is_solo = len(members) == 1

        leader = members_sorted[0] if members_sorted else None
        leader_strength = _calc_strength(leader) if leader else 0

        for pos, member in enumerate(members_sorted):
            info[member.car_number] = {
                "line_id": line_id,
                "line_pos": pos,  # 0=先頭, 1=番手, 2=3番手
                "line_size": len(members),
                "line_members": members_sorted,
                "is_solo": is_solo,
                "is_line_leader": pos == 0,
                "is_line_second": pos == 1,
                "line_avg_strength": line_avg,
                "line_top_strength": line_top,
                "line_leader_strength": leader_strength,
                "line_strength_std": float(np.std(line_strengths)) if len(line_strengths) > 1 else 0,
            }

    return info


def _style_order(line_group: str | None) -> int:
    """脚質をライン内の並び順に変換"""
    if not line_group:
        return 5
    g = line_group.strip()
    if g in ("逃", "先行"):
        return 0
    elif g in ("捲", "両", "自在"):
        return 1
    elif g in ("差",):
        return 2
    elif g in ("追", "マーク", "ク"):
        return 3
    return 5


def _calc_strength(entry: RaceEntry | None) -> float:
    """選手の総合力スコア"""
    if entry is None:
        return 0.0
    w = entry.win_rate or 0
    s = entry.second_rate or 0
    t = entry.third_rate or 0
    return (w * 3 + s * 2 + t) / 6


# =============================================================
# 直近走成績
# =============================================================

def _get_player_history(session: Session, player_id: int, current_race_id: int,
                        limit: int = 10) -> list[RaceEntry]:
    """選手の過去レース結果を取得（データリーク防止: 現在レースより前のみ）"""
    current_race = session.query(Race).filter_by(id=current_race_id).first()
    if not current_race:
        return []

    from sqlalchemy import or_, and_
    past_entries = (
        session.query(RaceEntry)
        .join(Race, RaceEntry.race_id == Race.id)
        .filter(
            RaceEntry.player_id == player_id,
            RaceEntry.finish_position.isnot(None),
            or_(
                Race.race_date < current_race.race_date,
                and_(
                    Race.race_date == current_race.race_date,
                    Race.id < current_race_id,
                ),
            ),
        )
        .order_by(Race.race_date.desc(), Race.id.desc())
        .limit(limit)
        .all()
    )
    return past_entries


def _calc_trend_features(history: list[RaceEntry]) -> dict:
    """直近走成績からトレンド特徴量を算出"""
    features = {
        "recent_races_count": len(history),
        "recent_avg_finish": 0.0,
        "recent_top3_rate": 0.0,
        "recent_win_rate": 0.0,
        "finish_trend_slope": 0.0,     # 傾き: 負=上り調子
        "recent_form_delta": 0.0,       # 直近 - 通算の差
    }

    if not history:
        return features

    positions = [e.finish_position for e in history if e.finish_position]
    if not positions:
        return features

    # 指数加重平均（直近を重視）
    weights = [0.9 ** i for i in range(len(positions))]
    ewm_avg = np.average(positions, weights=weights)

    features["recent_avg_finish"] = float(ewm_avg)
    features["recent_top3_rate"] = sum(1 for p in positions if p <= 3) / len(positions)
    features["recent_win_rate"] = sum(1 for p in positions if p == 1) / len(positions)

    # 着順の回帰傾きで調子判定（負=上り調子）
    if len(positions) >= 3:
        x = np.arange(len(positions), dtype=float)
        slope = np.polyfit(x, positions, 1)[0]
        features["finish_trend_slope"] = float(slope)

    return features


# =============================================================
# バンク × 脚質 相性
# =============================================================

def _calc_track_style_features(entry: RaceEntry, racecourse: Racecourse | None,
                                history: list[RaceEntry], session: Session) -> dict:
    """バンク長と脚質の相性特徴量"""
    track_len = racecourse.track_length if racecourse else 400
    style = (entry.line_group or "").strip()

    features = {
        "track_length": track_len,
        "is_short_track": 1 if track_len <= 335 else 0,   # 333mバンク
        "is_long_track": 1 if track_len >= 500 else 0,     # 500mバンク
        "style_num": _style_to_num(style),
        # 脚質×バンク交差特徴量
        "escape_on_short": 1 if style in ("逃", "先行") and track_len <= 335 else 0,
        "chase_on_long": 1 if style in ("差", "追", "マーク", "ク") and track_len >= 500 else 0,
    }

    # 同バンク長での過去成績
    if history and racecourse and session:
        same_track_results = []
        for h in history:
            h_race = session.query(Race).filter_by(id=h.race_id).first()
            if h_race:
                h_rc = session.query(Racecourse).filter_by(id=h_race.racecourse_id).first()
                if h_rc and h_rc.track_length == track_len and h.finish_position:
                    same_track_results.append(h.finish_position)

        if same_track_results:
            features["track_specific_avg"] = np.mean(same_track_results)
            features["track_specific_top3"] = sum(1 for p in same_track_results if p <= 3) / len(same_track_results)
        else:
            features["track_specific_avg"] = 5.0
            features["track_specific_top3"] = 0.3
    else:
        features["track_specific_avg"] = 5.0
        features["track_specific_top3"] = 0.3

    return features


def _style_to_num(style: str) -> int:
    """脚質を数値化"""
    mapping = {"逃": 1, "捲": 2, "両": 3, "自在": 3, "差": 4, "追": 5, "マーク": 5, "ク": 5}
    return mapping.get(style, 3)


# =============================================================
# 上がりタイム特徴量 (Phase 2)
# =============================================================

def _parse_finish_time(time_str: str | None) -> float | None:
    """finish_time文字列をfloatに変換"""
    if not time_str:
        return None
    try:
        return float(time_str)
    except (ValueError, TypeError):
        return None


def _calc_finish_time_features(entry: RaceEntry, race: Race,
                                racecourse: Racecourse | None,
                                history: list[RaceEntry],
                                session: Session) -> dict:
    """上がりタイムから脚力特徴量を算出"""
    features = {
        "finish_time_raw": 0.0,
        "finish_time_zscore": 0.0,       # バンク別標準化タイム
        "recent_avg_time": 0.0,           # 直近走平均タイム
        "time_improvement_slope": 0.0,    # タイム改善傾き（負=速くなっている）
        "time_vs_race_avg": 0.0,          # レース内平均との差
    }

    current_time = _parse_finish_time(entry.finish_time)

    # 直近走のタイム分析
    history_times = []
    for h in history:
        t = _parse_finish_time(h.finish_time)
        if t and 10.0 < t < 15.0:  # 妥当な範囲のみ
            history_times.append(t)

    if history_times:
        weights = [0.9 ** i for i in range(len(history_times))]
        features["recent_avg_time"] = float(np.average(history_times, weights=weights))

        # タイム改善傾き（負=速くなっている=良い）
        if len(history_times) >= 3:
            x = np.arange(len(history_times), dtype=float)
            slope = np.polyfit(x, history_times, 1)[0]
            features["time_improvement_slope"] = float(slope)

    # バンク別の標準化タイム（同バンクの過去結果と比較）
    if history and racecourse and session:
        track_len = racecourse.track_length or 400
        same_track_times = []
        for h in history:
            h_race = session.query(Race).filter_by(id=h.race_id).first()
            if h_race:
                h_rc = session.query(Racecourse).filter_by(id=h_race.racecourse_id).first()
                if h_rc and h_rc.track_length == track_len:
                    t = _parse_finish_time(h.finish_time)
                    if t and 10.0 < t < 15.0:
                        same_track_times.append(t)

        if len(same_track_times) >= 2:
            mean_t = np.mean(same_track_times)
            std_t = np.std(same_track_times)
            if std_t > 0 and features["recent_avg_time"] > 0:
                features["finish_time_zscore"] = float(
                    (features["recent_avg_time"] - mean_t) / std_t
                )

    return features


# =============================================================
# 決まり手特徴量 (Phase 2)
# =============================================================

def _calc_win_technique_features(entry: RaceEntry, history: list[RaceEntry],
                                  session: Session) -> dict:
    """選手の決まり手パターンから特徴量を算出"""
    features = {
        "technique_escape_rate": 0.0,    # 逃げ率
        "technique_scoop_rate": 0.0,     # 捲り率
        "technique_stretch_rate": 0.0,   # 差し率
        "technique_mark_rate": 0.0,      # マーク率
        "technique_diversity": 0.0,      # 決まり手の多様性
        "technique_matches_style": 0.0,  # 脚質と決まり手の一致度
    }

    if not history:
        return features

    # 過去の1〜2着時の決まり手を集計
    techniques = []
    for h in history:
        if h.win_technique and h.finish_position and h.finish_position <= 2:
            techniques.append(h.win_technique.strip())

    if not techniques:
        return features

    total = len(techniques)
    counts = {}
    for t in techniques:
        counts[t] = counts.get(t, 0) + 1

    features["technique_escape_rate"] = counts.get("逃", 0) / total
    features["technique_scoop_rate"] = counts.get("捲", 0) / total
    features["technique_stretch_rate"] = counts.get("差", 0) / total
    features["technique_mark_rate"] = counts.get("ク", 0) / total

    # 決まり手の多様性（エントロピー）
    probs = [c / total for c in counts.values() if c > 0]
    if probs:
        features["technique_diversity"] = float(-sum(p * np.log2(p) for p in probs if p > 0))

    # 脚質と決まり手の一致度
    style = (entry.line_group or "").strip()
    style_technique_map = {
        "逃": "逃",
        "追": "ク",
        "両": None,  # 自在はどれでもOK
    }
    expected = style_technique_map.get(style)
    if expected and expected in counts:
        features["technique_matches_style"] = counts[expected] / total
    elif style == "両":
        features["technique_matches_style"] = 0.5  # 自在は中立

    return features


# =============================================================
# 地元開催フラグ (Phase 3)
# =============================================================

# 都道府県の表記揺れ対応
_PREF_NORMALIZE = {
    "北海道": "北海道",
}


def _normalize_prefecture(pref: str | None) -> str:
    """都道府県の表記を統一（「県」「府」「都」を除去）"""
    if not pref:
        return ""
    p = pref.strip()
    # 「県」「府」「都」を除去して統一
    for suffix in ("県", "府", "都"):
        if p.endswith(suffix) and p != "京都":
            p = p[:-1]
            break
    return p


def _calc_hometown_features(player: Player | None,
                             racecourse: Racecourse | None) -> dict:
    """地元開催フラグを算出"""
    features = {
        "is_hometown": 0,
        "is_same_region": 0,
    }

    if not player or not racecourse:
        return features

    player_pref = _normalize_prefecture(player.prefecture)
    course_pref = _normalize_prefecture(racecourse.prefecture)

    if player_pref and course_pref:
        if player_pref == course_pref:
            features["is_hometown"] = 1
            features["is_same_region"] = 1
        else:
            # 地区判定（競輪の地区割り）
            regions = {
                "北日本": {"北海道", "青森", "秋田", "岩手", "山形", "宮城", "福島"},
                "関東": {"茨城", "栃木", "群馬", "埼玉", "東京", "千葉", "神奈川", "山梨", "長野", "新潟"},
                "南関東": {"埼玉", "東京", "千葉", "神奈川"},
                "中部": {"静岡", "愛知", "岐阜", "三重", "富山", "石川", "福井"},
                "近畿": {"滋賀", "京都", "大阪", "兵庫", "奈良", "和歌山"},
                "中国四国": {"鳥取", "島根", "岡山", "広島", "山口", "徳島", "香川", "愛媛", "高知"},
                "九州": {"福岡", "佐賀", "長崎", "熊本", "大分", "宮崎", "鹿児島", "沖縄"},
            }
            for region, prefs in regions.items():
                if player_pref in prefs and course_pref in prefs:
                    features["is_same_region"] = 1
                    break

    return features


# =============================================================
# 先行争い激化指数 (Phase 3)
# =============================================================

def _calc_pace_features(entries: list[RaceEntry]) -> dict:
    """レース内の先行争い激化指数を算出"""
    escape_count = 0
    scoop_count = 0
    for e in entries:
        style = (e.line_group or "").strip()
        if style in ("逃", "先行"):
            escape_count += 1
        elif style in ("捲", "両", "自在"):
            scoop_count += 1

    total = len(entries) if entries else 1
    return {
        "n_escape_riders": escape_count,
        "n_scoop_riders": scoop_count,
        "escape_ratio": escape_count / total,
        "pace_pressure": escape_count + scoop_count * 0.5,  # 先行争いの激しさ
    }


# =============================================================
# 年齢×トレンド交差特徴量 (Phase 3)
# =============================================================

def _calc_age_trend_features(player: Player | None,
                              trend_features: dict) -> dict:
    """年齢とトレンドの交差特徴量"""
    features = {
        "age_category": 2,          # 0=若手, 1=中堅, 2=ベテラン
        "young_growth": 0.0,        # 若手の成長指標
        "veteran_stability": 0.0,   # ベテランの安定性
        "age_trend_cross": 0.0,     # 年齢×トレンド交差
    }

    age = player.age if player and player.age else 30

    if age <= 25:
        features["age_category"] = 0
    elif age <= 35:
        features["age_category"] = 1
    else:
        features["age_category"] = 2

    slope = trend_features.get("finish_trend_slope", 0)
    recent_top3 = trend_features.get("recent_top3_rate", 0)

    # 若手×上り調子 = 成長ボーナス
    if age <= 25 and slope < 0:
        features["young_growth"] = abs(slope) * recent_top3

    # ベテラン×安定 = 安定ボーナス
    if age >= 35 and recent_top3 > 0.3:
        features["veteran_stability"] = recent_top3

    # 年齢×トレンド交差
    features["age_trend_cross"] = (40 - age) / 20.0 * (-slope) if slope != 0 else 0.0

    return features


# =============================================================
# レース格・距離特徴量 (v4)
# =============================================================

def _calc_race_context_features(race: Race, current_features: dict) -> dict:
    """レースの文脈（決勝/準決勝、距離）から特徴量を算出"""
    round_name = (race.round_name or "").strip()
    distance = race.distance or 0

    features = {
        "is_final": 1 if "決勝" in round_name and "準" not in round_name else 0,
        "is_semifinal": 1 if "準決" in round_name else 0,
        "is_first_day": 1 if "初日" in round_name or "1日" in round_name else 0,
        "distance": distance,
        # 決勝 × 地元 の交差（決勝での地元勢の爆発力）
        "final_x_hometown": 0,
        # 決勝 × グレード の交差
        "final_x_grade": 0,
    }

    if features["is_final"]:
        features["final_x_hometown"] = current_features.get("is_hometown", 0)
        features["final_x_grade"] = current_features.get("race_grade", 0)

    return features


# =============================================================
# ライン結束力 (v4)
# =============================================================

def _calc_line_cohesion(entry: RaceEntry, line_info: dict,
                         history: list[RaceEntry],
                         session: Session | None) -> dict:
    """同ラインメンバーとの過去のワンツー率等"""
    features = {
        "line_cohesion": 0.0,  # 同ラインメンバーとの過去連携率
        "same_region_line_ratio": 0.0,  # ライン内同地区率
    }

    if not line_info or not session:
        return features

    members = line_info.get("line_members", [])
    if len(members) <= 1:
        return features

    # ライン内の同地区率
    my_pref = ""
    if entry.player_id:
        from src.common.database import Player
        player = session.query(Player).filter_by(id=entry.player_id).first()
        if player:
            my_pref = _normalize_prefecture(player.prefecture)

    if my_pref and len(members) > 1:
        same_count = 0
        for m in members:
            if m.car_number == entry.car_number:
                continue
            if m.player_id:
                from src.common.database import Player
                mp = session.query(Player).filter_by(id=m.player_id).first()
                if mp and _normalize_prefecture(mp.prefecture) == my_pref:
                    same_count += 1
        features["same_region_line_ratio"] = same_count / (len(members) - 1)

    return features


# =============================================================
# 交互作用特徴量 (v4)
# =============================================================

def _calc_interaction_features(f: dict) -> dict:
    """既存特徴量の交差項"""
    return {
        # 番手 × 先頭の先行力
        "second_x_leader_escape": (
            f.get("is_line_second", 0) * f.get("line_leader_strength", 0)
        ),
        # ライン人数 × 先行争い激化
        "line_size_x_pace": (
            f.get("line_size", 1) * f.get("pace_pressure", 0)
        ),
        # 地元 × グレード
        "hometown_x_grade": (
            f.get("is_hometown", 0) * f.get("race_grade", 0)
        ),
        # 勝率 × 直近トレンド
        "winrate_x_trend": (
            f.get("win_rate", 0) * (1 - f.get("finish_trend_slope", 0))
        ),
        # 年齢 × 上がり改善
        "age_x_time_trend": (
            f.get("player_age", 30) * (-f.get("time_improvement_slope", 0))
        ),
    }


# =============================================================
# メイン特徴量構築
# =============================================================

def _build_entry_features(entry: RaceEntry, player: Player | None,
                           race: Race, racecourse: Racecourse | None,
                           all_entries: list[RaceEntry],
                           line_info: dict,
                           history: list[RaceEntry],
                           session: Session | None = None) -> dict:
    """1エントリーの全特徴量を生成"""
    features = {
        "car_number": entry.car_number,
        "frame_number": entry.frame_number or entry.car_number,
    }

    # === 選手基本成績 ===
    features["win_rate"] = entry.win_rate or 0.0
    features["second_rate"] = entry.second_rate or 0.0
    features["third_rate"] = entry.third_rate or 0.0
    features["avg_start"] = entry.avg_start or 0.0
    features["strength_score"] = _calc_strength(entry)

    # === 選手属性 ===
    if player:
        features["player_grade"] = _grade_to_num(player.grade)
        features["player_age"] = player.age or 30
    else:
        features["player_grade"] = 5
        features["player_age"] = 30

    # === レース情報 ===
    features["is_girl"] = 1 if race.is_girl else 0
    features["race_grade"] = _race_grade_to_num(race.grade)

    # === 車番有利度 ===
    features["car_number_advantage"] = _car_number_advantage(entry.car_number)

    # === コメント分析 ===
    features["comment_positive"] = _analyze_comment(entry.comment or "")

    # === ライン特徴量 (v2) ===
    li = line_info.get(entry.car_number, {})
    features["line_pos"] = li.get("line_pos", 2)
    features["line_size"] = li.get("line_size", 1)
    features["is_solo"] = 1 if li.get("is_solo", True) else 0
    features["is_line_leader"] = 1 if li.get("is_line_leader", False) else 0
    features["is_line_second"] = 1 if li.get("is_line_second", False) else 0
    features["line_avg_strength"] = li.get("line_avg_strength", features["strength_score"])
    features["line_top_strength"] = li.get("line_top_strength", features["strength_score"])
    features["line_leader_strength"] = li.get("line_leader_strength", 0)
    features["line_strength_std"] = li.get("line_strength_std", 0)
    # 番手が先頭より強いか（番手捲り狙いの指標）
    if li.get("is_line_second") and li.get("line_leader_strength", 0) > 0:
        features["second_vs_leader"] = features["strength_score"] - li["line_leader_strength"]
    else:
        features["second_vs_leader"] = 0.0

    # === 直近走トレンド (v2) ===
    trend = _calc_trend_features(history)
    features.update(trend)
    # 直近成績と通算成績の乖離
    if features["win_rate"] > 0:
        features["recent_form_delta"] = trend["recent_top3_rate"] - (features["third_rate"] / 100)
    else:
        features["recent_form_delta"] = 0.0

    # === バンク × 脚質 (v2) ===
    track_features = _calc_track_style_features(entry, racecourse, history, session)
    features.update(track_features)

    # === 上がりタイム (v3) ===
    time_features = _calc_finish_time_features(entry, race, racecourse, history, session)
    features.update(time_features)

    # === 決まり手 (v3) ===
    technique_features = _calc_win_technique_features(entry, history, session)
    features.update(technique_features)

    # === 地元開催 (v3) ===
    hometown_features = _calc_hometown_features(player, racecourse)
    features.update(hometown_features)

    # === 年齢×トレンド (v3) ===
    age_trend = _calc_age_trend_features(player, trend)
    features.update(age_trend)

    # === レース格・距離 (v4) ===
    features.update(_calc_race_context_features(race, features))

    # === ライン結束力 (v4) ===
    features.update(_calc_line_cohesion(entry, li, history, session))

    # === 交互作用特徴量 (v4) ===
    features.update(_calc_interaction_features(features))

    # === ターゲット ===
    if entry.finish_position is not None:
        features["finish_position"] = entry.finish_position

    return features


# =============================================================
# レース内相対特徴量
# =============================================================

def _add_relative_features(df: pd.DataFrame) -> pd.DataFrame:
    """レース内での相対的な位置を特徴量化"""
    if df.empty:
        return df

    # 勝率のレース内順位とz-score
    if "win_rate" in df.columns:
        df["win_rate_rank"] = df["win_rate"].rank(ascending=False, method="min")
        mean_wr = df["win_rate"].mean()
        std_wr = df["win_rate"].std()
        df["win_rate_z"] = (df["win_rate"] - mean_wr) / std_wr if std_wr > 0 else 0
        df["win_rate_diff_from_top"] = df["win_rate"].max() - df["win_rate"]

    # 総合力のレース内順位
    if "strength_score" in df.columns:
        df["strength_rank"] = df["strength_score"].rank(ascending=False, method="min")
        mean_ss = df["strength_score"].mean()
        std_ss = df["strength_score"].std()
        df["strength_z"] = (df["strength_score"] - mean_ss) / std_ss if std_ss > 0 else 0

    # ライン強度のレース内順位
    if "line_avg_strength" in df.columns:
        df["line_strength_rank"] = df["line_avg_strength"].rank(ascending=False, method="min")

    # 級班差（最強との差）
    if "player_grade" in df.columns:
        df["grade_gap_to_best"] = df["player_grade"] - df["player_grade"].min()

    return df


# =============================================================
# ユーティリティ
# =============================================================

def _grade_to_num(grade: str | None) -> int:
    mapping = {"SS": 1, "S1": 2, "S2": 3, "A1": 4, "A2": 5, "A3": 6, "L1": 4}
    return mapping.get(grade or "", 5)


def _race_grade_to_num(grade: str | None) -> int:
    mapping = {"GP": 7, "GI": 6, "GII": 5, "GIII": 4, "FI": 3, "FII": 2}
    if not grade:
        return 1
    for key, val in mapping.items():
        if key in (grade or "").upper():
            return val
    return 1


def _car_number_advantage(car_number: int) -> float:
    advantage = {1: 1.0, 2: 0.9, 3: 0.85, 4: 0.8, 5: 0.75, 6: 0.7, 7: 0.65, 8: 0.6, 9: 0.55}
    return advantage.get(car_number, 0.5)


def _analyze_comment(comment: str) -> float:
    if not comment:
        return 0.0

    positive_words = [
        "好調", "調子いい", "仕上が", "自信", "積極", "攻め", "万全",
        "状態いい", "いい感じ", "脚の感じ", "軽い", "切れ", "伸び",
        "しっかり", "ガンガン", "全力", "思い切り", "勝ち", "優勝",
        "決勝", "気合", "手応え", "良い", "よい",
    ]
    negative_words = [
        "不調", "痛", "怪我", "落車", "心配", "不安", "重い",
        "戻って", "まだ", "厳しい", "課題", "ダメ", "だめ",
        "回復", "違和感", "疲", "休み明け",
    ]

    score = 0.0
    for w in positive_words:
        if w in comment:
            score += 0.15
    for w in negative_words:
        if w in comment:
            score -= 0.2

    return max(-1.0, min(1.0, score))


# 特徴量カラム（モデル入力に使用するもの）
FEATURE_COLUMNS = [
    # 基本成績
    "frame_number", "win_rate", "second_rate", "third_rate",
    "avg_start", "strength_score",
    # 選手属性
    "player_grade", "player_age",
    # レース情報
    "is_girl", "race_grade", "car_number_advantage",
    # コメント
    "comment_positive",
    # ライン特徴量 (v2)
    "line_pos", "line_size", "is_solo", "is_line_leader", "is_line_second",
    "line_avg_strength", "line_top_strength", "line_leader_strength",
    "line_strength_std", "second_vs_leader",
    # 直近走トレンド (v2)
    "recent_races_count", "recent_avg_finish", "recent_top3_rate",
    "recent_win_rate", "finish_trend_slope", "recent_form_delta",
    # バンク × 脚質 (v2)
    "track_length", "is_short_track", "is_long_track", "style_num",
    "escape_on_short", "chase_on_long",
    "track_specific_avg", "track_specific_top3",
    # レース内相対 (v2)
    "win_rate_rank", "win_rate_z", "win_rate_diff_from_top",
    "strength_rank", "strength_z",
    "line_strength_rank", "grade_gap_to_best",
    # 上がりタイム (v3)
    "finish_time_raw", "finish_time_zscore",
    "recent_avg_time", "time_improvement_slope", "time_vs_race_avg",
    # 決まり手 (v3)
    "technique_escape_rate", "technique_scoop_rate",
    "technique_stretch_rate", "technique_mark_rate",
    "technique_diversity", "technique_matches_style",
    # 地元開催 (v3)
    "is_hometown", "is_same_region",
    # 先行争い激化指数 (v3)
    "n_escape_riders", "n_scoop_riders", "escape_ratio", "pace_pressure",
    # 年齢×トレンド (v3)
    "age_category", "young_growth", "veteran_stability", "age_trend_cross",
    # レース格・距離 (v4)
    "is_final", "is_semifinal", "is_first_day", "distance",
    "final_x_hometown", "final_x_grade",
    # ライン結束力 (v4)
    "line_cohesion", "same_region_line_ratio",
    # 交互作用 (v4)
    "second_x_leader_escape", "line_size_x_pace",
    "hometown_x_grade", "winrate_x_trend", "age_x_time_trend",
]
