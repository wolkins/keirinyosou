"""スクレイピングデータをDBに格納するパーサー"""
from datetime import datetime

from sqlalchemy.orm import Session

from src.common.database import Odds, Player, Race, RaceEntry, Racecourse, get_session


# Kドリームス slug → 競輪場名のマッピング
SLUG_TO_NAME = {
    "hakodate": "函館", "aomori": "青森", "iwakidaira": "いわき平",
    "yahiko": "弥彦", "maebashi": "前橋", "toride": "取手",
    "utsunomiya": "宇都宮", "omiya": "大宮", "seibu-en": "西武園",
    "keiokaku": "京王閣", "tachikawa": "立川", "matsudo": "松戸",
    "chiba": "千葉", "kawasaki": "川崎", "hiratsuka": "平塚",
    "odawara": "小田原", "ito": "伊東温泉", "shizuoka": "静岡",
    "nagoya": "名古屋", "gifu": "岐阜", "ogaki": "大垣",
    "toyohashi": "豊橋", "toyama": "富山", "matsusaka": "松阪",
    "yokkaichi": "四日市", "fukui": "福井", "nara": "奈良",
    "mukomachi": "向日町", "wakayama": "和歌山", "kishiwada": "岸和田",
    "tamano": "玉野", "hiroshima": "広島", "hofu": "防府",
    "takamatsu": "高松", "komatsushima": "小松島", "kochi": "高知",
    "matsuyama": "松山", "kokura": "小倉", "kurume": "久留米",
    "takeo": "武雄", "sasebo": "佐世保", "beppu": "別府",
    "kumamoto": "熊本",
}


def store_race_entries(session: Session, race_date: str, venue_code: str,
                       race_number: int, detail: dict, race_name: str = "",
                       grade: str = "", venue_slug: str = "") -> Race | None:
    """レース＋出走表をDBに格納

    Args:
        session: DB Session
        race_date: "YYYY-MM-DD"
        venue_code: 競輪場コード
        race_number: レース番号
        detail: scrape_race_detail() の返却値
        race_name: レース名
        grade: グレード
        venue_slug: 競輪場のURLスラッグ (コード不一致時のフォールバック)

    Returns:
        作成/更新されたRaceオブジェクト
    """
    # 競輪場を取得 (slug優先 → コードでフォールバック)
    racecourse = None
    if venue_slug:
        venue_name = SLUG_TO_NAME.get(venue_slug)
        if venue_name:
            racecourse = session.query(Racecourse).filter_by(name=venue_name).first()
    if not racecourse:
        racecourse = session.query(Racecourse).filter_by(code=venue_code).first()
    if not racecourse:
        print(f"  [WARN] 競輪場 code={venue_code} slug={venue_slug} が見つかりません")
        return None

    dt = datetime.strptime(race_date, "%Y-%m-%d").date()

    # Raceの upsert
    race = session.query(Race).filter_by(
        race_date=dt, racecourse_id=racecourse.id, race_number=race_number
    ).first()

    race_info = detail.get("race", {})
    if not race:
        race = Race(
            race_date=dt,
            racecourse_id=racecourse.id,
            race_number=race_number,
            race_name=race_name or race_info.get("race_name", ""),
            grade=grade or race_info.get("grade", ""),
            distance=race_info.get("distance"),
            status="scheduled",
        )
        session.add(race)
        session.flush()
    else:
        if race_name:
            race.race_name = race_name
        if grade:
            race.grade = grade
        if race_info.get("distance"):
            race.distance = race_info["distance"]

    # エントリーを格納
    entries = detail.get("entries", [])
    for e in entries:
        car_number = e.get("car_number")
        if car_number is None:
            continue

        # 選手の upsert
        player_id_str = str(e.get("player_id", "")).strip()
        player_name = e.get("player_name", "").strip()
        player = None

        if player_id_str:
            player = session.query(Player).filter_by(player_id=player_id_str).first()
        elif player_name:
            # IDが無い場合は名前で検索
            player = session.query(Player).filter_by(name=player_name).first()

        if player is None and (player_id_str or player_name):
            player = Player(
                player_id=player_id_str or f"name_{player_name}",
                name=player_name,
                prefecture=e.get("prefecture", ""),
                grade=e.get("grade", ""),
            )
            session.add(player)
            session.flush()
        elif player:
            if player_name:
                player.name = player_name
            if e.get("grade"):
                player.grade = e["grade"]
            if e.get("prefecture"):
                player.prefecture = e["prefecture"]

        # RaceEntry の upsert
        entry = session.query(RaceEntry).filter_by(
            race_id=race.id, car_number=car_number
        ).first()

        if not entry:
            entry = RaceEntry(
                race_id=race.id,
                player_id=player.id if player else None,
                frame_number=e.get("frame_number") or car_number,
                car_number=car_number,
                line_group=e.get("line_group", ""),
                line_label=e.get("line_label", ""),
                win_rate=e.get("win_rate"),
                second_rate=e.get("second_rate"),
                third_rate=e.get("third_rate"),
                avg_start=e.get("avg_start"),
                comment=e.get("comment", ""),
            )
            session.add(entry)
        else:
            entry.win_rate = e.get("win_rate") or entry.win_rate
            entry.second_rate = e.get("second_rate") or entry.second_rate
            entry.third_rate = e.get("third_rate") or entry.third_rate
            entry.comment = e.get("comment") or entry.comment
            entry.line_group = e.get("line_group") or entry.line_group
            entry.line_label = e.get("line_label") or entry.line_label

    session.commit()
    return race


def store_odds(session: Session, race: Race, odds_list: list[dict]):
    """オッズをDBに格納"""
    for o in odds_list:
        if not o.get("combination") or o.get("odds_value") is None:
            continue

        existing = session.query(Odds).filter_by(
            race_id=race.id,
            bet_type=o.get("bet_type", ""),
            combination=o["combination"],
        ).first()

        if not existing:
            session.add(Odds(
                race_id=race.id,
                bet_type=o.get("bet_type", ""),
                combination=o["combination"],
                odds_value=o["odds_value"],
            ))
        else:
            existing.odds_value = o["odds_value"]
            existing.captured_at = datetime.now()

    session.commit()


def store_results(session: Session, race: Race, result_data: dict):
    """レース結果をDBに格納"""
    results = result_data.get("results", [])
    for r in results:
        car_number = r.get("car_number")
        if car_number is None:
            continue

        entry = session.query(RaceEntry).filter_by(
            race_id=race.id, car_number=car_number
        ).first()

        if entry:
            entry.finish_position = r.get("finish_position")
            entry.finish_time = r.get("finish_time", "")
            entry.win_technique = r.get("win_technique", "")

    race.status = "finished"
    session.commit()
