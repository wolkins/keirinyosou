"""アプリケーション設定"""
from pathlib import Path

# プロジェクトルート
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# データベース
DB_PATH = PROJECT_ROOT / "data" / "keirin.db"
DB_URL = f"sqlite:///{DB_PATH}"

# スクレイピング設定
REQUEST_DELAY = 3.0   # リクエスト間隔(秒)
DAY_PAUSE = 30.0      # 日ごとの休憩(秒)
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36"
)

# モデル設定
DECAY_HALF_LIFE_DAYS = 90  # 時間減衰の半減期(日)
