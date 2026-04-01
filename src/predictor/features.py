"""特徴量エンジニアリング

各選手エントリーからモデル入力用の特徴量を生成する。
"""
import re

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from src.common.database import Odds, Player, Race, RaceEntry, Racecourse


def build_features_for_race(session: Session, race: Race) -> pd.DataFrame:
    """レースの全エントリーから特徴量DataFrameを生成

    Returns:
        columns: car_number, + 各特徴量, + target (finish_position, あれば)
    """
    entries = session.query(RaceEntry).filter_by(race_id=race.id).all()
    racecourse = session.query(Racecourse).filter_by(id=race.racecourse_id).first()

    rows = []
    for entry in entries:
        player = session.query(Player).filter_by(id=entry.player_id).first() if entry.player_id else None
        row = _build_entry_features(entry, player, race, racecourse, entries)
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    return df


def _build_entry_features(entry: RaceEntry, player: Player | None,
                           race: Race, racecourse: Racecourse | None,
                           all_entries: list[RaceEntry]) -> dict:
    """1エントリーの特徴量を生成"""
    features = {
        "car_number": entry.car_number,
        "frame_number": entry.frame_number or entry.car_number,
    }

    # === 選手基本成績 ===
    features["win_rate"] = entry.win_rate or 0.0
    features["second_rate"] = entry.second_rate or 0.0
    features["third_rate"] = entry.third_rate or 0.0
    features["avg_start"] = entry.avg_start or 0.0

    # 成績の強さスコア
    features["strength_score"] = (
        (features["win_rate"] * 3 + features["second_rate"] * 2 + features["third_rate"]) / 6
    )

    # === 選手属性 ===
    if player:
        features["player_grade"] = _grade_to_num(player.grade)
        features["player_age"] = player.age or 30
    else:
        features["player_grade"] = 5  # 不明
        features["player_age"] = 30

    # === レース情報 ===
    features["track_length"] = racecourse.track_length if racecourse else 400
    features["is_girl"] = 1 if race.is_girl else 0
    features["race_grade"] = _race_grade_to_num(race.grade)

    # === ライン情報 ===
    features["line_position"] = _line_position_to_num(entry.line_group)

    # 同ラインの選手数
    same_line_count = sum(
        1 for e in all_entries
        if e.line_label and entry.line_label and e.line_label == entry.line_label
    )
    features["line_size"] = same_line_count

    # 同ラインの平均強さ
    same_line_entries = [
        e for e in all_entries
        if e.line_label and entry.line_label and e.line_label == entry.line_label
    ]
    if same_line_entries:
        features["line_avg_strength"] = np.mean([
            ((e.win_rate or 0) * 3 + (e.second_rate or 0) * 2 + (e.third_rate or 0)) / 6
            for e in same_line_entries
        ])
    else:
        features["line_avg_strength"] = features["strength_score"]

    # === 車番有利度（統計的にインが有利） ===
    features["car_number_advantage"] = _car_number_advantage(entry.car_number)

    # === コメント分析（簡易版）===
    features["comment_positive"] = _analyze_comment(entry.comment or "")

    # === 他選手との相対的な位置 ===
    all_win_rates = [e.win_rate or 0 for e in all_entries if e.win_rate is not None]
    if all_win_rates:
        features["win_rate_rank_in_race"] = sorted(all_win_rates, reverse=True).index(features["win_rate"]) + 1
        features["win_rate_diff_from_top"] = max(all_win_rates) - features["win_rate"]
    else:
        features["win_rate_rank_in_race"] = 5
        features["win_rate_diff_from_top"] = 0

    # === ターゲット（結果がある場合）===
    if entry.finish_position is not None:
        features["finish_position"] = entry.finish_position

    return features


def _grade_to_num(grade: str | None) -> int:
    """選手の級班を数値に変換（小さいほど強い）"""
    mapping = {"SS": 1, "S1": 2, "S2": 3, "A1": 4, "A2": 5, "A3": 6, "L1": 4}
    return mapping.get(grade or "", 5)


def _race_grade_to_num(grade: str | None) -> int:
    """レースグレードを数値に変換（大きいほど高グレード）"""
    mapping = {"GP": 7, "GI": 6, "GII": 5, "GIII": 4, "FI": 3, "FII": 2}
    if not grade:
        return 1
    for key, val in mapping.items():
        if key in (grade or "").upper():
            return val
    return 1


def _line_position_to_num(line_group: str | None) -> int:
    """ライン位置を数値に変換"""
    if not line_group:
        return 2
    lg = line_group.strip()
    if "先行" in lg or "自在" in lg:
        return 1
    elif "番手" in lg:
        return 2
    elif "3番手" in lg or "三番手" in lg:
        return 3
    return 2


def _car_number_advantage(car_number: int) -> float:
    """車番の有利度（統計に基づく近似値）"""
    # 1番車が最も有利、外に行くほど不利
    advantage = {1: 1.0, 2: 0.9, 3: 0.85, 4: 0.8, 5: 0.75, 6: 0.7, 7: 0.65, 8: 0.6, 9: 0.55}
    return advantage.get(car_number, 0.5)


def _analyze_comment(comment: str) -> float:
    """選手コメントの簡易感情分析

    ポジティブ語・ネガティブ語の出現で判定。
    Returns: -1.0 ~ 1.0
    """
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
    "frame_number",
    "win_rate",
    "second_rate",
    "third_rate",
    "avg_start",
    "strength_score",
    "player_grade",
    "player_age",
    "track_length",
    "is_girl",
    "race_grade",
    "line_position",
    "line_size",
    "line_avg_strength",
    "car_number_advantage",
    "comment_positive",
    "win_rate_rank_in_race",
    "win_rate_diff_from_top",
]
