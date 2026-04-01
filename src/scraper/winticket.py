"""Winticket スクレイパー

WinticketはSPA (React + Redux)で、HTMLに __PRELOADED_STATE__ としてデータが埋め込まれている。
HTMLをフェッチしてJSONを抽出する方式でデータを取得する。
"""
import json
import re
from datetime import date, datetime

from bs4 import BeautifulSoup

from .base import BaseScraper

BASE_URL = "https://www.winticket.jp"

# Winticket競輪場slug → コードのマッピング
VENUE_SLUGS = {
    "hakodate": "01", "aomori": "02", "iwakidaira": "03", "yahiko": "04",
    "maebashi": "05", "toride": "06", "utsunomiya": "07", "omiya": "08",
    "seibu-en": "09", "keiokaku": "10", "tachikawa": "11", "matsudo": "13",
    "chiba": "14", "kawasaki": "15", "hiratsuka": "16", "odawara": "17",
    "ito-onsen": "18", "shizuoka": "19", "nagoya": "20", "gifu": "21",
    "ogaki": "22", "toyohashi": "23", "toyama": "24", "matsusaka": "25",
    "yokkaichi": "27", "fukui": "31", "nara": "34", "muko-machi": "35",
    "wakayama": "36", "kishiwada": "37", "tamano": "42", "hiroshima": "43",
    "hofu": "45", "takamatsu": "46", "komatsushima": "47", "kochi": "51",
    "matsuyama": "52", "kokura": "54", "kurume": "55", "takeo": "56",
    "sasebo": "57", "beppu": "58", "kumamoto": "59",
}

SLUG_TO_CODE = VENUE_SLUGS
CODE_TO_SLUG = {v: k for k, v in VENUE_SLUGS.items()}


class WinticketScraper(BaseScraper):
    """ウィンチケットからのデータ取得"""

    def _extract_preloaded_state(self, html: str) -> dict | None:
        """HTML内の __PRELOADED_STATE__ JSONを抽出"""
        pattern = r'window\.__PRELOADED_STATE__\s*=\s*(\{.+?\});?\s*</script>'
        match = re.search(pattern, html, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        return None

    def _extract_tanstack_data(self, html: str) -> dict | None:
        """HTML内の TanStack Query dehydrated state を抽出"""
        pattern = r'window\.__REACT_QUERY_STATE__\s*=\s*(\{.+?\});?\s*</script>'
        match = re.search(pattern, html, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        return None

    def scrape_schedule(self, year: int, month: int) -> list[dict]:
        """月間スケジュールを取得

        Returns:
            [{"date": "2026-04-01", "venue_slug": "keiokaku", "venue_code": "10", ...}, ...]
        """
        url = f"{BASE_URL}/keirin/schedules/{year:04d}{month:02d}"
        resp = self._get(url)
        state = self._extract_preloaded_state(resp.text)

        schedules = []
        if state:
            # state構造からスケジュールデータを探索
            schedules = self._parse_schedule_state(state)

        # stateが取れない場合はHTMLパースにフォールバック
        if not schedules:
            schedules = self._parse_schedule_html(resp.text)

        return schedules

    def _parse_schedule_state(self, state: dict) -> list[dict]:
        """Reduxステートからスケジュールを解析"""
        schedules = []
        # state内を再帰的に探索してスケジュールデータを見つける
        for key in state:
            if "schedule" in key.lower() or "race" in key.lower():
                data = state[key]
                if isinstance(data, dict):
                    for sub_key, sub_val in data.items():
                        if isinstance(sub_val, list):
                            for item in sub_val:
                                if isinstance(item, dict) and ("date" in item or "dateVenueId" in item):
                                    schedules.append(item)
        return schedules

    def _parse_schedule_html(self, html: str) -> list[dict]:
        """HTMLからスケジュールをパース (フォールバック)"""
        soup = BeautifulSoup(html, "lxml")
        schedules = []

        # スケジュールページのリンクからレース情報を抽出
        links = soup.find_all("a", href=re.compile(r"/keirin/\w+/racecard/"))
        for link in links:
            href = link.get("href", "")
            match = re.search(r"/keirin/(\w[\w-]*)/racecard/(\d+)/(\d+)/(\d+)", href)
            if match:
                venue_slug = match.group(1)
                date_venue_id = match.group(2)
                day = match.group(3)
                race_num = match.group(4)
                schedules.append({
                    "venue_slug": venue_slug,
                    "venue_code": SLUG_TO_CODE.get(venue_slug, ""),
                    "date_venue_id": date_venue_id,
                    "day": day,
                    "race_number": int(race_num),
                    "url": href,
                })

        return schedules

    def scrape_race_list(self, date_str: str) -> list[dict]:
        """指定日のレース一覧を取得

        Args:
            date_str: "YYYY-MM-DD"

        Returns:
            [{"venue_slug": str, "venue_code": str, "date_venue_id": str,
              "day": str, "race_number": int, "race_name": str, ...}, ...]
        """
        url = f"{BASE_URL}/keirin/racecard"
        resp = self._get(url)

        races = []
        state = self._extract_preloaded_state(resp.text)
        if state:
            races = self._parse_racelist_state(state, date_str)

        if not races:
            races = self._parse_racelist_html(resp.text, date_str)

        return races

    def _parse_racelist_state(self, state: dict, date_str: str) -> list[dict]:
        """Reduxステートからレース一覧を解析"""
        races = []
        # 深い構造の中からレースデータを探す
        self._find_races_recursive(state, races, date_str)
        return races

    def _find_races_recursive(self, obj, races: list, date_str: str, depth: int = 0):
        """再帰的にレースデータを探索"""
        if depth > 5 or not isinstance(obj, (dict, list)):
            return
        if isinstance(obj, list):
            for item in obj:
                self._find_races_recursive(item, races, date_str, depth + 1)
        elif isinstance(obj, dict):
            # レースっぽいデータかチェック
            if "raceNumber" in obj or "race_number" in obj:
                races.append(obj)
            else:
                for val in obj.values():
                    self._find_races_recursive(val, races, date_str, depth + 1)

    def _parse_racelist_html(self, html: str, date_str: str) -> list[dict]:
        """HTMLからレース一覧をパース"""
        soup = BeautifulSoup(html, "lxml")
        races = []

        links = soup.find_all("a", href=re.compile(r"/keirin/[\w-]+/racecard/"))
        for link in links:
            href = link.get("href", "")
            match = re.search(r"/keirin/([\w-]+)/racecard/(\d+)/(\d+)/(\d+)", href)
            if match:
                venue_slug = match.group(1)
                date_venue_id = match.group(2)
                day = match.group(3)
                race_num = int(match.group(4))

                race_name_el = link.find(class_=re.compile(r"race.*name|title", re.I))
                race_name = race_name_el.get_text(strip=True) if race_name_el else ""

                races.append({
                    "venue_slug": venue_slug,
                    "venue_code": SLUG_TO_CODE.get(venue_slug, ""),
                    "date_venue_id": date_venue_id,
                    "day": day,
                    "race_number": race_num,
                    "race_name": race_name,
                    "race_date": date_str,
                    "url": f"{BASE_URL}{href}",
                })

        return races

    def scrape_race_detail(self, race_info: dict) -> dict:
        """レース詳細（出走表）を取得

        Args:
            race_info: scrape_race_list()の返却値の1要素

        Returns:
            {"race": {...}, "entries": [{...}, ...]}
        """
        venue_slug = race_info["venue_slug"]
        dvid = race_info["date_venue_id"]
        day = race_info["day"]
        rno = race_info["race_number"]

        url = f"{BASE_URL}/keirin/{venue_slug}/racecard/{dvid}/{day}/{rno}"
        resp = self._get(url)

        state = self._extract_preloaded_state(resp.text)
        if state:
            result = self._parse_racecard_state(state, race_info)
            if result and result.get("entries"):
                return result

        return self._parse_racecard_html(resp.text, race_info)

    def _parse_racecard_state(self, state: dict, race_info: dict) -> dict:
        """Reduxステートから出走表を解析"""
        entries = []
        race_data = {}

        # stateの中からエントリーデータを探す
        def find_entries(obj, depth=0):
            if depth > 6 or not isinstance(obj, (dict, list)):
                return
            if isinstance(obj, list):
                for item in obj:
                    find_entries(item, depth + 1)
            elif isinstance(obj, dict):
                # 選手エントリーっぽいデータ
                if any(k in obj for k in ("playerName", "player_name", "frameNumber", "frame_number", "carNumber", "car_number")):
                    entries.append(obj)
                else:
                    # レース情報っぽいデータ
                    if any(k in obj for k in ("gradeName", "grade_name", "raceName", "race_name")):
                        race_data.update(obj)
                    for val in obj.values():
                        find_entries(val, depth + 1)

        find_entries(state)

        parsed_entries = []
        for e in entries:
            parsed = {
                "car_number": e.get("carNumber") or e.get("car_number"),
                "frame_number": e.get("frameNumber") or e.get("frame_number"),
                "player_id": str(e.get("playerId") or e.get("player_id") or e.get("registrationNumber") or ""),
                "player_name": e.get("playerName") or e.get("player_name") or e.get("name", ""),
                "grade": e.get("grade") or e.get("playerGrade") or "",
                "prefecture": e.get("prefecture") or e.get("area") or "",
                "win_rate": self._safe_float(e.get("winRate") or e.get("win_rate")),
                "second_rate": self._safe_float(e.get("secondRate") or e.get("second_rate") or e.get("twoRate")),
                "third_rate": self._safe_float(e.get("thirdRate") or e.get("third_rate") or e.get("threeRate")),
                "avg_start": self._safe_float(e.get("avgStartPosition") or e.get("avg_start")),
                "comment": e.get("comment") or e.get("playerComment") or "",
                "line_group": e.get("lineGroup") or e.get("line_group") or "",
                "line_label": e.get("lineLabel") or e.get("line_label") or "",
            }
            if parsed["car_number"] is not None:
                parsed_entries.append(parsed)

        return {
            "race": {
                "race_name": race_data.get("raceName") or race_data.get("race_name") or race_info.get("race_name", ""),
                "grade": race_data.get("gradeName") or race_data.get("grade_name") or race_info.get("grade", ""),
                "distance": race_data.get("distance") or race_data.get("trackLength"),
            },
            "entries": parsed_entries,
        }

    def _parse_racecard_html(self, html: str, race_info: dict) -> dict:
        """HTMLから出走表をパース (フォールバック)"""
        soup = BeautifulSoup(html, "lxml")
        entries = []

        # 出走表のテーブルを探す
        # Winticketの出走表は選手カード形式で表示される
        player_cards = soup.find_all(
            ["div", "li", "tr"],
            class_=re.compile(r"player|racer|entry|cyclist", re.I)
        )

        for card in player_cards:
            entry = self._extract_entry_from_html(card)
            if entry and entry.get("car_number"):
                entries.append(entry)

        # 選手カードが見つからない場合、テーブル行を試す
        if not entries:
            tables = soup.find_all("table")
            for table in tables:
                rows = table.find_all("tr")
                for row in rows:
                    entry = self._extract_entry_from_table_row(row)
                    if entry and entry.get("car_number"):
                        entries.append(entry)

        return {
            "race": {
                "race_name": race_info.get("race_name", ""),
                "grade": race_info.get("grade", ""),
                "distance": None,
            },
            "entries": entries,
        }

    def _extract_entry_from_html(self, element) -> dict:
        """HTML要素から選手エントリーを抽出"""
        text = element.get_text(" ", strip=True)
        entry = {
            "car_number": None,
            "frame_number": None,
            "player_id": "",
            "player_name": "",
            "grade": "",
            "prefecture": "",
            "win_rate": None,
            "second_rate": None,
            "third_rate": None,
            "avg_start": None,
            "comment": "",
            "line_group": "",
            "line_label": "",
        }

        # 車番を数字で探す
        num_el = element.find(class_=re.compile(r"number|car|waku|frame", re.I))
        if num_el:
            num_text = num_el.get_text(strip=True)
            m = re.search(r"(\d+)", num_text)
            if m:
                entry["car_number"] = int(m.group(1))

        # 選手名
        name_el = element.find(class_=re.compile(r"name|player|racer", re.I))
        if name_el:
            entry["player_name"] = name_el.get_text(strip=True)

        # 勝率等の数値
        rate_els = element.find_all(class_=re.compile(r"rate|percentage|ratio", re.I))
        rates = []
        for rel in rate_els:
            m = re.search(r"(\d+\.?\d*)", rel.get_text(strip=True))
            if m:
                rates.append(float(m.group(1)))
        if len(rates) >= 3:
            entry["win_rate"] = rates[0]
            entry["second_rate"] = rates[1]
            entry["third_rate"] = rates[2]

        # コメント
        comment_el = element.find(class_=re.compile(r"comment|message", re.I))
        if comment_el:
            entry["comment"] = comment_el.get_text(strip=True)

        return entry

    def _extract_entry_from_table_row(self, row) -> dict:
        """テーブル行から選手エントリーを抽出"""
        cells = row.find_all(["td", "th"])
        if len(cells) < 3:
            return {}

        entry = {
            "car_number": None,
            "frame_number": None,
            "player_id": "",
            "player_name": "",
            "grade": "",
            "prefecture": "",
            "win_rate": None,
            "second_rate": None,
            "third_rate": None,
            "avg_start": None,
            "comment": "",
            "line_group": "",
            "line_label": "",
        }

        for i, cell in enumerate(cells):
            text = cell.get_text(strip=True)
            if i == 0:
                m = re.search(r"(\d+)", text)
                if m:
                    entry["car_number"] = int(m.group(1))
            elif re.match(r"^[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]+$", text):
                if not entry["player_name"]:
                    entry["player_name"] = text
            elif re.match(r"^\d+\.?\d*$", text):
                val = float(text)
                if entry["win_rate"] is None:
                    entry["win_rate"] = val
                elif entry["second_rate"] is None:
                    entry["second_rate"] = val
                elif entry["third_rate"] is None:
                    entry["third_rate"] = val

        return entry

    def scrape_odds(self, race_info: dict) -> list[dict]:
        """オッズを取得

        Returns:
            [{"bet_type": str, "combination": str, "odds_value": float}, ...]
        """
        venue_slug = race_info["venue_slug"]
        dvid = race_info["date_venue_id"]
        day = race_info["day"]
        rno = race_info["race_number"]

        url = f"{BASE_URL}/keirin/{venue_slug}/odds/{dvid}/{day}/{rno}"
        resp = self._get(url)

        state = self._extract_preloaded_state(resp.text)
        odds_list = []

        if state:
            odds_list = self._parse_odds_state(state)

        if not odds_list:
            odds_list = self._parse_odds_html(resp.text)

        return odds_list

    def _parse_odds_state(self, state: dict) -> list[dict]:
        """Reduxステートからオッズを解析"""
        odds = []

        def find_odds(obj, depth=0):
            if depth > 6 or not isinstance(obj, (dict, list)):
                return
            if isinstance(obj, list):
                for item in obj:
                    find_odds(item, depth + 1)
            elif isinstance(obj, dict):
                if "odds" in obj and "combination" in obj:
                    odds.append({
                        "bet_type": obj.get("betType") or obj.get("bet_type", ""),
                        "combination": str(obj.get("combination", "")),
                        "odds_value": self._safe_float(obj.get("odds")),
                    })
                elif "oddsValue" in obj or "odds_value" in obj:
                    odds.append({
                        "bet_type": obj.get("betType") or obj.get("bet_type", ""),
                        "combination": str(obj.get("combination") or obj.get("number", "")),
                        "odds_value": self._safe_float(obj.get("oddsValue") or obj.get("odds_value")),
                    })
                else:
                    for val in obj.values():
                        find_odds(val, depth + 1)

        find_odds(state)
        return odds

    def _parse_odds_html(self, html: str) -> list[dict]:
        """HTMLからオッズをパース"""
        soup = BeautifulSoup(html, "lxml")
        odds = []

        # オッズテーブルを探す
        tables = soup.find_all("table")
        for table in tables:
            rows = table.find_all("tr")
            for row in rows:
                cells = row.find_all(["td", "th"])
                if len(cells) >= 2:
                    combo_text = cells[0].get_text(strip=True)
                    odds_text = cells[-1].get_text(strip=True)

                    combo_match = re.search(r"(\d[\d\-=]+\d)", combo_text)
                    odds_match = re.search(r"(\d+\.?\d*)", odds_text)

                    if combo_match and odds_match:
                        odds.append({
                            "bet_type": "",
                            "combination": combo_match.group(1),
                            "odds_value": float(odds_match.group(1)),
                        })

        return odds

    def scrape_result(self, race_info: dict) -> dict:
        """レース結果を取得

        Returns:
            {"results": [{"car_number": int, "finish_position": int, "finish_time": str,
                          "win_technique": str}, ...],
             "payouts": [{"bet_type": str, "combination": str, "payout": int}, ...]}
        """
        venue_slug = race_info["venue_slug"]
        dvid = race_info["date_venue_id"]
        day = race_info["day"]
        rno = race_info["race_number"]

        # 結果ページ（出走表と同じURLだが結果後はデータが更新される）
        url = f"{BASE_URL}/keirin/{venue_slug}/racecard/{dvid}/{day}/{rno}"
        resp = self._get(url)

        state = self._extract_preloaded_state(resp.text)
        if state:
            result = self._parse_result_state(state)
            if result and result.get("results"):
                return result

        return self._parse_result_html(resp.text)

    def _parse_result_state(self, state: dict) -> dict:
        """Reduxステートからレース結果を解析"""
        results = []
        payouts = []

        def find_results(obj, depth=0):
            if depth > 6 or not isinstance(obj, (dict, list)):
                return
            if isinstance(obj, list):
                for item in obj:
                    find_results(item, depth + 1)
            elif isinstance(obj, dict):
                if "finishPosition" in obj or "finish_position" in obj or "ranking" in obj:
                    results.append({
                        "car_number": obj.get("carNumber") or obj.get("car_number"),
                        "finish_position": obj.get("finishPosition") or obj.get("finish_position") or obj.get("ranking"),
                        "finish_time": str(obj.get("finishTime") or obj.get("finish_time") or obj.get("goalTime", "")),
                        "win_technique": obj.get("winTechnique") or obj.get("win_technique") or obj.get("decidingFactor", ""),
                    })
                elif "payout" in obj or "payoff" in obj:
                    payouts.append({
                        "bet_type": obj.get("betType") or obj.get("bet_type", ""),
                        "combination": str(obj.get("combination") or obj.get("number", "")),
                        "payout": obj.get("payout") or obj.get("payoff") or 0,
                    })
                else:
                    for val in obj.values():
                        find_results(val, depth + 1)

        find_results(state)
        return {"results": results, "payouts": payouts}

    def _parse_result_html(self, html: str) -> dict:
        """HTMLから結果をパース"""
        soup = BeautifulSoup(html, "lxml")
        results = []

        # 結果テーブルを探す
        result_els = soup.find_all(
            ["div", "tr", "li"],
            class_=re.compile(r"result|ranking|finish", re.I)
        )

        for i, el in enumerate(result_els):
            text = el.get_text(" ", strip=True)
            car_match = re.search(r"(\d)\s*番", text)
            if car_match:
                results.append({
                    "car_number": int(car_match.group(1)),
                    "finish_position": i + 1,
                    "finish_time": "",
                    "win_technique": "",
                })

        return {"results": results, "payouts": []}

    @staticmethod
    def _safe_float(val) -> float | None:
        """安全にfloat変換"""
        if val is None:
            return None
        try:
            return float(val)
        except (ValueError, TypeError):
            return None
