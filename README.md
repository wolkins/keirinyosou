# keirinyosou

Kドリームスのデータをスクレイピングし、LightGBM LambdaRank で競輪レース着順を予測するCLIツール。

## セットアップ

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
mkdir -p data
```

## CLI

```bash
# データ取得（単日）
python -m src.cli.main scrape --date 2026-04-01 [--with-results] [--with-odds]

# データ取得（範囲）
python -m src.cli.main scrape-range --from 2025-04-01 [--to 2026-03-31] [--with-results]

# 予測
python -m src.cli.main predict --date 2026-04-01 [--venue 川口] [--mode accuracy|roi]

# 学習
python -m src.cli.main train [--min-races 50]

# DB状況確認
python -m src.cli.main status
```

## Web UI

```bash
python -m src.cli.main serve [--port 8000] [--reload]
```

FastAPI + Jinja2 + htmx。http://localhost:8000 でアクセス。

## サーバー運用

```bash
# バックグラウンドでスクレイピング
nohup .venv/bin/python -m src.cli.main scrape-range --from 2025-04-01 --with-results > scrape.log 2>&1 &

# ログ確認
tail -f scrape.log

# Web UIをバックグラウンド起動
nohup .venv/bin/python -m src.cli.main serve --port 8000 > serve.log 2>&1 &
```

## 構成

```
src/
  scraper/       Kドリームスからスクレイピング
  parser/store.py  DB格納
  predictor/
    features.py  64特徴量
    model.py     LightGBM LambdaRank
  cli/main.py    CLIエントリポイント
  web/           FastAPI + Jinja2 + htmx
data/keirin.db   SQLite
```
