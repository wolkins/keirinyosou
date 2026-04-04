#!/bin/bash
# 前日分のデータを自動取得する日次スクリプト
cd "$(dirname "$0")"
YESTERDAY=$(date -d "yesterday" +%Y-%m-%d)
source .venv/bin/activate
python -m src.cli.main scrape --date "$YESTERDAY" --with-results >> logs/daily_scrape.log 2>&1
