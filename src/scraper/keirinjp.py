"""KEIRIN.JP (keirin.jp) スクレイパー

dataplazaエンドポイントを使用してレースデータを取得する。
Winticketのバックアップデータソースとして利用。
"""
import json
import re
from datetime import datetime

from bs4 import BeautifulSoup

from .base import BaseScraper

BASE_URL = "https://keirin.jp"


class KeirinJpScraper(BaseScraper):
    """KEIRIN.JP からのデータ取得"""

    def scrape_race_list(self, date_str: str) -> list[dict]:
        """指定日のレース一覧を取得

        Args:
            date_str: "YYYY-MM-DD"

        Returns:
            [{"venue_code": str, "venue_name": str, "race_date": str,
              "race_number": int, "grade": str}, ...]
        """
        # トップページから開催情報のJSONを取得
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        url = f"{BASE_URL}/pc/top"
        resp = self._get(url)

        races = []

        # pc0101_json から開催情報を抽出
        match = re.search(r'var\s+pc0101_json\s*=\s*(\[.+?\]);', resp.text, re.DOTALL)
        if match:
            try:
                schedule_data = json.loads(match.group(1))
                for item in schedule_data:
                    if isinstance(item, dict):
                        item_date = item.get("kaisai_bi", "")
                        # YYYYMMDD形式の日付を比較
                        target_date = dt.strftime("%Y%m%d")
                        if item_date == target_date or str(item.get("date", "")) == target_date:
                            venue_code = str(item.get("jyo_cd", item.get("kcd", "")))
                            venue_name = item.get("jyo_name", item.get("jyo_nm", ""))
                            grade = item.get("grade_name", item.get("grade", ""))

                            # 各レース番号を追加
                            max_race = int(item.get("max_race", 12))
                            for rno in range(1, max_race + 1):
                                races.append({
                                    "venue_code": venue_code.zfill(2),
                                    "venue_name": venue_name,
                                    "race_date": date_str,
                                    "race_number": rno,
                                    "grade": grade,
                                    "source": "keirinjp",
                                })
            except json.JSONDecodeError:
                pass

        return races

    def scrape_race_detail(self, race_info: dict) -> dict:
        """dataplazaの出走表から選手データを取得"""
        venue_code = race_info["venue_code"]
        date_str = race_info["race_date"].replace("-", "")
        race_num = race_info["race_number"]

        url = f"{BASE_URL}/pc/dfw/dataplaza/guest/raceresult"
        params = {"KCD": venue_code, "KBI": date_str, "RNO": str(race_num)}

        resp = self._get(url, params=params)
        soup = BeautifulSoup(resp.text, "lxml")

        entries = []
        # dataplazaのテーブルをパース
        tables = soup.find_all("table")
        for table in tables:
            rows = table.find_all("tr")
            for row in rows[1:]:  # ヘッダー行をスキップ
                cells = row.find_all(["td", "th"])
                if len(cells) >= 6:
                    entry = self._parse_dataplaza_row(cells)
                    if entry.get("car_number"):
                        entries.append(entry)

        race_name_el = soup.find(class_=re.compile(r"race.*name|heading", re.I))
        race_name = race_name_el.get_text(strip=True) if race_name_el else ""

        return {
            "race": {
                "race_name": race_name,
                "grade": race_info.get("grade", ""),
                "distance": None,
            },
            "entries": entries,
        }

    def _parse_dataplaza_row(self, cells) -> dict:
        """dataplazaテーブル行をパース"""
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
            elif i == 1:
                # 選手名（リンク内にある場合）
                link = cell.find("a")
                if link:
                    entry["player_name"] = link.get_text(strip=True)
                    href = link.get("href", "")
                    pid_m = re.search(r"player_id=(\d+)", href)
                    if pid_m:
                        entry["player_id"] = pid_m.group(1)
                else:
                    entry["player_name"] = text

        # 数値データ（勝率等）を後方のセルから抽出
        numbers = []
        for cell in cells[2:]:
            text = cell.get_text(strip=True)
            m = re.match(r"^(\d+\.?\d*)$", text)
            if m:
                numbers.append(float(m.group(1)))

        if len(numbers) >= 3:
            entry["win_rate"] = numbers[0]
            entry["second_rate"] = numbers[1]
            entry["third_rate"] = numbers[2]

        return entry

    def scrape_odds(self, race_info: dict) -> list[dict]:
        """KEIRIN.JPからオッズ取得（現在未実装 - Winticketを優先）"""
        return []

    def scrape_result(self, race_info: dict) -> dict:
        """dataplazaのレース結果を取得"""
        venue_code = race_info["venue_code"]
        date_str = race_info["race_date"].replace("-", "")
        race_num = race_info["race_number"]

        url = f"{BASE_URL}/pc/dfw/dataplaza/guest/raceresult"
        params = {"KCD": venue_code, "KBI": date_str, "RNO": str(race_num)}

        resp = self._get(url, params=params)
        soup = BeautifulSoup(resp.text, "lxml")

        results = []
        payouts = []

        # 着順テーブルを探す
        tables = soup.find_all("table")
        for table in tables:
            caption = table.find("caption")
            header_row = table.find("tr")
            header_text = header_row.get_text(" ", strip=True) if header_row else ""

            if "着順" in header_text or "着" in (caption.get_text() if caption else ""):
                rows = table.find_all("tr")[1:]
                for pos, row in enumerate(rows, 1):
                    cells = row.find_all("td")
                    if len(cells) >= 2:
                        car_text = cells[0].get_text(strip=True)
                        car_m = re.search(r"(\d+)", car_text)
                        if car_m:
                            results.append({
                                "car_number": int(car_m.group(1)),
                                "finish_position": pos,
                                "finish_time": cells[-1].get_text(strip=True) if len(cells) > 2 else "",
                                "win_technique": "",
                            })

            # 払戻テーブル
            if "払戻" in header_text or "配当" in header_text:
                rows = table.find_all("tr")[1:]
                for row in rows:
                    cells = row.find_all("td")
                    if len(cells) >= 3:
                        bet_type = cells[0].get_text(strip=True)
                        combination = cells[1].get_text(strip=True)
                        payout_text = cells[2].get_text(strip=True).replace(",", "").replace("円", "")
                        payout_m = re.search(r"(\d+)", payout_text)
                        if payout_m:
                            payouts.append({
                                "bet_type": bet_type,
                                "combination": combination,
                                "payout": int(payout_m.group(1)),
                            })

        return {"results": results, "payouts": payouts}
