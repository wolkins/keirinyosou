"""スクレイパー基底クラス"""
import time
from abc import ABC, abstractmethod

import requests

from src.common.config import REQUEST_DELAY, USER_AGENT


class BaseScraper(ABC):
    """スクレイパーの基底クラス"""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        self._last_request_time = 0.0

    def _get(self, url: str, params: dict | None = None) -> requests.Response:
        """レート制限付きGETリクエスト"""
        elapsed = time.time() - self._last_request_time
        if elapsed < REQUEST_DELAY:
            time.sleep(REQUEST_DELAY - elapsed)

        resp = self.session.get(url, params=params, timeout=30)
        self._last_request_time = time.time()
        resp.raise_for_status()
        return resp

    @abstractmethod
    def scrape_race_list(self, date_str: str) -> list[dict]:
        """指定日のレース一覧を取得"""
        ...

    @abstractmethod
    def scrape_race_detail(self, race_info: dict) -> dict:
        """レース詳細（出走表）を取得"""
        ...

    @abstractmethod
    def scrape_odds(self, race_info: dict) -> list[dict]:
        """オッズを取得"""
        ...

    @abstractmethod
    def scrape_result(self, race_info: dict) -> dict:
        """レース結果を取得"""
        ...
