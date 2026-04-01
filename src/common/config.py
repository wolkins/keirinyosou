"""アプリケーション設定"""
from pathlib import Path

# プロジェクトルート
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# データベース
DB_PATH = PROJECT_ROOT / "data" / "keirin.db"
DB_URL = f"sqlite:///{DB_PATH}"

# スクレイピング設定
REQUEST_DELAY = 2.0  # リクエスト間隔(秒)
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36"
)
