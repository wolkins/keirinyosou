"""楽天Kドリームス (keirin.kdreams.jp) スクレイパー

レースID体系:
  [場コード2桁][年4桁][月2桁][日2桁][開催回2桁][日目2桁][レース番号4桁]
  例: 2720260401010001 = 京王閣(27)/2026年04月01日/第01回/01日目/第0001R
"""
import re
from datetime import datetime

from bs4 import BeautifulSoup

from .base import BaseScraper

BASE_URL = "https://keirin.kdreams.jp"


class KdreamsScraper(BaseScraper):
    """楽天Kドリームスからのデータ取得"""

    def scrape_race_list(self, date_str: str) -> list[dict]:
        """指定日のレース一覧を取得"""
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        url = f"{BASE_URL}/racecard/{dt.year}/{dt.month:02d}/{dt.day:02d}/"
        resp = self._get(url)
        soup = BeautifulSoup(resp.text, "lxml")

        races = []
        seen = set()

        # racedetail リンクを収集
        links = soup.find_all("a", href=re.compile(r"/[\w-]+/racedetail/\d+/"))
        for link in links:
            href = link.get("href", "")
            match = re.search(r"/([\w-]+)/racedetail/(\d{14,18})/", href)
            if not match:
                continue

            venue_slug = match.group(1)
            race_id = match.group(2)

            if race_id in seen:
                continue
            seen.add(race_id)

            parsed = self._parse_race_id(race_id)
            if not parsed:
                continue

            races.append({
                "venue_code": parsed["venue_code"],
                "venue_slug": venue_slug,
                "race_date": date_str,
                "race_number": parsed["race_number"],
                "race_id": race_id,
                "date_venue_id": race_id[:10],
                "day": str(parsed["day"]),
                "race_name": link.get_text(strip=True),
                "grade": "",
                "source": "kdreams",
            })

        races.sort(key=lambda r: (r["venue_code"], r["race_number"]))
        return races

    def scrape_race_detail(self, race_info: dict) -> dict:
        """出走表を取得（BeautifulSoupで直接パース）"""
        venue_slug = race_info["venue_slug"]
        race_id = race_info["race_id"]

        url = f"{BASE_URL}/{venue_slug}/racedetail/{race_id}/"
        resp = self._get(url)
        soup = BeautifulSoup(resp.text, "lxml")

        entries = self._parse_entries(soup)

        # レース名
        title_el = soup.find("title")
        race_name = ""
        if title_el:
            # "宇都宮競輪 レース詳細 | 〇〇杯 1R Ａ級予選 | ..." から抽出
            parts = title_el.get_text().split("|")
            if len(parts) >= 2:
                race_name = parts[1].strip()

        # 距離
        distance = None
        dist_match = re.search(r"(\d{3,4})\s*[mM㍍メートル]", resp.text)
        if dist_match:
            distance = int(dist_match.group(1))

        return {
            "race": {
                "race_name": race_name or race_info.get("race_name", ""),
                "grade": race_info.get("grade", ""),
                "distance": distance,
            },
            "entries": entries,
        }

    def _parse_entries(self, soup: BeautifulSoup) -> list[dict]:
        """出走表テーブルから選手データを抽出（最初のテーブルのみ）

        ページ内に複数の予想ビューがあるが、データが完全な最初のテーブルのみ使う。
        """
        entries = []
        seen_car_numbers = set()

        # class="n1", "n2" ... の行を探す
        rows = soup.find_all("tr", class_=re.compile(r"^n\d+"))
        for row in rows:
            entry = self._parse_entry_row(row)
            if entry and entry.get("car_number"):
                # 重複排除: 同じ車番は最初に見つかったもの（勝率付き）のみ
                if entry["car_number"] not in seen_car_numbers:
                    seen_car_numbers.add(entry["car_number"])
                    entries.append(entry)

        return entries

    def _parse_entry_row(self, row) -> dict | None:
        """1行分の選手データをパース"""
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

        # 枠番: td.bracket
        bracket_td = row.find("td", class_="bracket")
        if bracket_td:
            text = bracket_td.get_text(strip=True)
            m = re.search(r"(\d+)", text)
            if m:
                entry["frame_number"] = int(m.group(1))

        # 車番: td.num
        num_td = row.find("td", class_="num")
        if num_td:
            text = num_td.get_text(strip=True)
            m = re.search(r"(\d+)", text)
            if m:
                entry["car_number"] = int(m.group(1))

        if entry["car_number"] is None:
            return None

        # 選手名・府県: td.rider
        rider_td = row.find("td", class_="rider")
        if rider_td:
            # 選手名はテキスト直下、府県はspan.home内
            texts = rider_td.get_text("\n", strip=True).split("\n")
            if texts:
                entry["player_name"] = texts[0].strip()
            home_span = rider_td.find("span", class_="home")
            if home_span:
                home_text = home_span.get_text(strip=True)
                # "埼　玉/45/88" → 府県, 年齢, 期
                parts = re.split(r"[/／]", home_text)
                if parts:
                    entry["prefecture"] = parts[0].replace("　", "").strip()

            # 選手リンクからIDを抽出
            player_link = rider_td.find("a")
            if player_link:
                href = player_link.get("href", "")
                pid_m = re.search(r"/(\d{4,6})/", href)
                if pid_m:
                    entry["player_id"] = pid_m.group(1)
                # リンクテキストが選手名
                entry["player_name"] = player_link.get_text(strip=True) or entry["player_name"]

        # 全tdを取得して後方の数値データを解析
        all_tds = row.find_all("td")
        # 後ろ3つが 勝率, 2連対率, 3連対率
        numeric_values = []
        for td in all_tds:
            text = td.get_text(strip=True)
            m = re.match(r"^(\d+\.\d+)$", text)
            if m:
                numeric_values.append(float(m.group(1)))

        # 最後の3つの小数値が 勝率, 2連対率, 3連対率
        if len(numeric_values) >= 3:
            entry["win_rate"] = numeric_values[-3]
            entry["second_rate"] = numeric_values[-2]
            entry["third_rate"] = numeric_values[-1]

        # 級班: riderの次のtd
        if rider_td:
            next_td = rider_td.find_next_sibling("td")
            if next_td:
                grade_text = next_td.get_text(strip=True)
                if grade_text in ("SS", "S1", "S2", "A1", "A2", "A3", "L1"):
                    entry["grade"] = grade_text

        # 脚質（ライン情報として使用）
        all_td_texts = [td.get_text(strip=True) for td in all_tds]
        for text in all_td_texts:
            if text in ("逃", "捲", "差", "追", "両"):
                entry["line_group"] = text
                break

        return entry

    def scrape_odds(self, race_info: dict) -> list[dict]:
        """オッズ取得（Winticketの方がデータ豊富なため未実装）"""
        return []

    def scrape_result(self, race_info: dict) -> dict:
        """レース結果を取得

        結果テーブル (table.result_table):
          Row0: ヘッダー [予想, 着順, 車番, 選手名, 着差, 上り, 決まり手, S/B, 勝敗因]
          Row1+: [◎, 1, 2, 尾崎 悠生, , 12.0, 逃, B, ]

        払戻テーブル (table.refund_table):
          [2枠連複, 未発売, 2車連複, 2=5 170円(1), 3連勝複, 2=5=6 650円(3), ワイド, ...]
        """
        venue_slug = race_info["venue_slug"]
        race_id = race_info["race_id"]

        url = f"{BASE_URL}/{venue_slug}/racedetail/{race_id}/"
        resp = self._get(url)
        soup = BeautifulSoup(resp.text, "lxml")

        results = []
        payouts = []

        # 着順テーブル
        result_table = soup.find("table", class_="result_table")
        if result_table:
            rows = result_table.find_all("tr")[1:]  # ヘッダーをスキップ
            for row in rows:
                cells = row.find_all("td")
                if len(cells) < 4:
                    continue
                # cells: [予想, 着順, 車番, 選手名, 着差, 上り, 決まり手, S/B, 勝敗因]
                pos_text = cells[1].get_text(strip=True)
                car_text = cells[2].get_text(strip=True)
                technique = cells[6].get_text(strip=True) if len(cells) > 6 else ""
                finish_time = cells[5].get_text(strip=True) if len(cells) > 5 else ""

                pos_m = re.search(r"(\d+)", pos_text)
                car_m = re.search(r"(\d+)", car_text)
                if pos_m and car_m:
                    results.append({
                        "car_number": int(car_m.group(1)),
                        "finish_position": int(pos_m.group(1)),
                        "finish_time": finish_time,
                        "win_technique": technique,
                    })

        # 払戻テーブル (dt=組み合わせ, dd=金額)
        refund_table = soup.find("table", class_="refund_table")
        if refund_table:
            for dl in refund_table.find_all("dl"):
                dt = dl.find("dt")
                dd = dl.find("dd")
                if dt and dd:
                    combo = dt.get_text(strip=True)
                    amount_text = dd.get_text(strip=True)
                    amount_m = re.search(r"([\d,]+)円", amount_text)
                    if combo and amount_m and combo != "未発売":
                        payouts.append({
                            "bet_type": "",
                            "combination": combo,
                            "payout": int(amount_m.group(1).replace(",", "")),
                        })

        return {"results": results, "payouts": payouts}

    @staticmethod
    def _parse_race_id(race_id: str) -> dict | None:
        """レースIDをパース

        例: 2720260401010001
        → venue_code=27, date=20260401, round=01, day=01, race_number=1
        """
        if len(race_id) < 16:
            return None
        try:
            return {
                "venue_code": race_id[:2],
                "date": race_id[2:10],
                "round": int(race_id[10:12]),
                "day": int(race_id[12:14]),
                "race_number": int(race_id[14:18]),
            }
        except (ValueError, IndexError):
            return None
