#!/bin/bash
# 競輪: 学習 + さくらVPSへモデル転送
cd "$(dirname "$0")"
source .venv/bin/activate
echo "学習開始: $(date)"
python -m src.cli.main train
echo "学習完了: $(date)"
echo "モデル転送中..."
scp -P 3843 data/models/*.pkl ubuntu@153.126.161.119:/data/work/keirinyosou/data/models/
echo "転送完了: $(date)"
