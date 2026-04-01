"""予測モデル v2 - LightGBM LambdaRank

Codex/Gemini提案を統合:
- LightGBM LambdaRank でレース内ランキング学習
- 同じモデルで予測し、賭け判断ロジックで的中率/回収率を切替
- 時系列CV（ランダム分割禁止）
- ケリー基準での資金配分（回収率モード）
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except (ImportError, OSError):
    HAS_LIGHTGBM = False
    from sklearn.ensemble import GradientBoostingRegressor

from src.common.config import PROJECT_ROOT
from src.common.database import Odds, Race, RaceEntry, get_session

from .features import FEATURE_COLUMNS, build_features_for_race

MODEL_DIR = PROJECT_ROOT / "data" / "models"


class KeirinPredictor:
    """競輪予測モデル（LightGBM LambdaRank）"""

    def __init__(self, mode: str = "accuracy"):
        """
        Args:
            mode: "accuracy" (的中率重視) or "roi" (回収率重視)
        """
        self.mode = mode
        self.model = None
        self._model_path = MODEL_DIR / "model_ranker.pkl"

    def train(self, session: Session, min_races: int = 50) -> dict:
        """過去のレース結果からLambdaRankモデルを学習

        時系列分割: 最後20%を検証データに使用
        """
        races = (
            session.query(Race)
            .filter(Race.status == "finished")
            .order_by(Race.race_date)
            .all()
        )
        if len(races) < min_races:
            return {"error": f"学習データが不足しています ({len(races)}/{min_races}レース)"}

        # 特徴量を構築
        all_dfs = []
        race_ids = []
        for race in races:
            df = build_features_for_race(session, race)
            if not df.empty and "finish_position" in df.columns:
                df["race_id"] = race.id
                all_dfs.append(df)
                race_ids.append(race.id)

        if not all_dfs:
            return {"error": "有効なトレーニングデータがありません"}

        full_df = pd.concat(all_dfs, ignore_index=True)

        # ラベル: 着順を反転（1着=9点, 9着=1点）
        max_pos = full_df["finish_position"].max()
        full_df["label"] = (max_pos + 1 - full_df["finish_position"]).clip(lower=0)

        # 時系列分割（最後20%を検証用）
        unique_races = full_df["race_id"].unique()
        split_idx = int(len(unique_races) * 0.8)
        train_race_ids = set(unique_races[:split_idx])
        valid_race_ids = set(unique_races[split_idx:])

        train_df = full_df[full_df["race_id"].isin(train_race_ids)].sort_values("race_id")
        valid_df = full_df[full_df["race_id"].isin(valid_race_ids)].sort_values("race_id")

        available_cols = [c for c in FEATURE_COLUMNS if c in train_df.columns]

        X_train = train_df[available_cols].fillna(0)
        y_train = train_df["label"]
        group_train = train_df.groupby("race_id").size().tolist()

        X_valid = valid_df[available_cols].fillna(0)
        y_valid = valid_df["label"]
        group_valid = valid_df.groupby("race_id").size().tolist()

        # モデル学習
        if HAS_LIGHTGBM:
            ranker = lgb.LGBMRanker(
                objective="lambdarank",
                metric="ndcg",
                ndcg_eval_at=[1, 3],
                learning_rate=0.05,
                num_leaves=63,
                min_data_in_leaf=max(20, len(train_df) // 100),
                feature_fraction=0.8,
                n_estimators=1000,
                verbose=-1,
            )
            callbacks = [lgb.early_stopping(50, verbose=False)]
            if len(valid_df) > 0 and group_valid:
                ranker.fit(
                    X_train, y_train, group=group_train,
                    eval_set=[(X_valid, y_valid)],
                    eval_group=[group_valid],
                    callbacks=callbacks,
                )
            else:
                ranker.fit(X_train, y_train, group=group_train)
        else:
            # フォールバック: sklearn GradientBoosting（着順回帰）
            ranker = GradientBoostingRegressor(
                n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42,
            )
            ranker.fit(X_train, y_train)

        self.model = ranker

        # モデル保存
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        with open(self._model_path, "wb") as f:
            pickle.dump({"model": ranker, "features": available_cols}, f)

        # 評価指標
        eval_results = {}
        if len(valid_df) > 0:
            valid_df = valid_df.copy()
            valid_df["pred_score"] = ranker.predict(X_valid)
            # Top1的中率
            top1_hits = 0
            top3_hits = 0
            n_eval_races = 0
            for rid in valid_race_ids:
                race_data = valid_df[valid_df["race_id"] == rid]
                if race_data.empty:
                    continue
                n_eval_races += 1
                pred_top = race_data.nlargest(1, "pred_score")["finish_position"].values
                if len(pred_top) > 0 and pred_top[0] == 1:
                    top1_hits += 1
                pred_top3_cars = set(race_data.nlargest(3, "pred_score")["car_number"].values)
                actual_top3_cars = set(race_data.nsmallest(3, "finish_position")["car_number"].values)
                if pred_top3_cars == actual_top3_cars:
                    top3_hits += 1

            if n_eval_races > 0:
                eval_results["top1_accuracy"] = top1_hits / n_eval_races
                eval_results["top3_exact"] = top3_hits / n_eval_races

        return {
            "n_races": len(races),
            "n_train_races": len(train_race_ids),
            "n_valid_races": len(valid_race_ids),
            "n_samples": len(full_df),
            "n_features": len(available_cols),
            **eval_results,
        }

    def load(self) -> bool:
        """保存済みモデルを読み込み"""
        if self._model_path.exists():
            with open(self._model_path, "rb") as f:
                data = pickle.load(f)
            if isinstance(data, dict):
                self.model = data["model"]
                self._feature_cols = data.get("features", FEATURE_COLUMNS)
            else:
                self.model = data
                self._feature_cols = FEATURE_COLUMNS
            return True
        return False

    def predict(self, session: Session, race: Race) -> list[dict]:
        """レースの予想を行う"""
        if self.model is None:
            if not self.load():
                return self._predict_by_stats(session, race)

        df = build_features_for_race(session, race)
        if df.empty:
            return []

        feature_cols = getattr(self, "_feature_cols", FEATURE_COLUMNS)
        available_cols = [c for c in feature_cols if c in df.columns]
        X = df[available_cols].fillna(0)

        # LambdaRankスコア
        scores = self.model.predict(X)

        # スコアを0-1に正規化
        s_min, s_max = scores.min(), scores.max()
        if s_max > s_min:
            norm_scores = (scores - s_min) / (s_max - s_min)
        else:
            norm_scores = np.ones_like(scores) * 0.5

        # オッズ情報
        odds_data = session.query(Odds).filter_by(race_id=race.id, bet_type="win").all()
        odds_map = {}
        for o in odds_data:
            try:
                odds_map[int(o.combination)] = o.odds_value
            except ValueError:
                pass

        # エントリー情報
        entries = session.query(RaceEntry).filter_by(race_id=race.id).all()
        entry_map = {e.car_number: e for e in entries}

        results = []
        for i, row in df.iterrows():
            car_num = int(row["car_number"])
            entry = entry_map.get(car_num)
            score = float(norm_scores[i])
            raw_score = float(scores[i])

            odds_val = odds_map.get(car_num, 0)

            # 期待値 = 予測確率 × オッズ
            expected_value = score * odds_val if odds_val else None

            # モードによるスコア
            if self.mode == "roi" and expected_value is not None:
                final_score = expected_value
            else:
                final_score = score

            results.append({
                "car_number": car_num,
                "player_name": entry.player.name if entry and entry.player else "不明",
                "grade": entry.player.grade if entry and entry.player else "",
                "win_rate": entry.win_rate or 0 if entry else 0,
                "second_rate": entry.second_rate or 0 if entry else 0,
                "third_rate": entry.third_rate or 0 if entry else 0,
                "score": round(final_score, 4),
                "probability": round(score, 4),
                "raw_score": round(raw_score, 4),
                "odds": odds_val,
                "expected_value": round(expected_value, 4) if expected_value else None,
                "recommendation": self._get_recommendation(score),
                "comment": entry.comment or "" if entry else "",
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    def _predict_by_stats(self, session: Session, race: Race) -> list[dict]:
        """モデルが無い場合の統計ベース予測"""
        entries = session.query(RaceEntry).filter_by(race_id=race.id).all()
        if not entries:
            return []

        results = []
        for entry in entries:
            win_r = entry.win_rate or 0
            sec_r = entry.second_rate or 0
            thi_r = entry.third_rate or 0

            grade_bonus = {"SS": 10, "S1": 7, "S2": 5, "A1": 3, "A2": 1, "A3": 0}.get(
                entry.player.grade if entry.player else "", 0
            )

            car_bonus = max(0, (6 - entry.car_number)) * 0.5
            comment_score = _analyze_comment_for_stats(entry.comment) * 5

            score = (win_r * 3 + sec_r * 2 + thi_r) / 6 + grade_bonus + car_bonus + comment_score

            odds_data = session.query(Odds).filter_by(
                race_id=race.id, bet_type="win", combination=str(entry.car_number)
            ).first()
            odds_val = odds_data.odds_value if odds_data else 0

            results.append({
                "car_number": entry.car_number,
                "player_name": entry.player.name if entry and entry.player else "不明",
                "grade": entry.player.grade if entry and entry.player else "",
                "win_rate": win_r,
                "second_rate": sec_r,
                "third_rate": thi_r,
                "score": round(score, 4),
                "probability": round(score / 100, 4),
                "raw_score": round(score, 4),
                "odds": odds_val,
                "expected_value": round(score / 100 * odds_val, 4) if odds_val else None,
                "recommendation": self._get_recommendation(score / 100),
                "comment": entry.comment or "",
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    def _get_recommendation(self, score: float) -> str:
        if score >= 0.7:
            return "◎ 本命"
        elif score >= 0.5:
            return "○ 対抗"
        elif score >= 0.35:
            return "▲ 単穴"
        elif score >= 0.2:
            return "△ 連下"
        else:
            return "×"

    def suggest_bets(self, predictions: list[dict], budget: int = 1000) -> list[dict]:
        """推奨買い目を生成"""
        if not predictions:
            return []

        top3 = predictions[:3]
        bets = []

        if self.mode == "accuracy":
            # 的中率重視: 本命を厚く
            bets.append({
                "bet_type": "2車単",
                "combination": f"{top3[0]['car_number']}-{top3[1]['car_number']}",
                "amount": budget // 3,
                "reason": f"本命 {top3[0]['player_name']} → 対抗 {top3[1]['player_name']}",
            })
            if len(top3) >= 3:
                bets.append({
                    "bet_type": "2車単",
                    "combination": f"{top3[0]['car_number']}-{top3[2]['car_number']}",
                    "amount": budget // 4,
                    "reason": f"本命 {top3[0]['player_name']} → 単穴 {top3[2]['player_name']}",
                })
            bets.append({
                "bet_type": "2車複",
                "combination": f"{top3[0]['car_number']}={top3[1]['car_number']}",
                "amount": budget // 4,
                "reason": f"上位2名: {top3[0]['player_name']}, {top3[1]['player_name']}",
            })

        else:
            # 回収率重視: 期待値ベース + ケリー基準
            by_ev = sorted(
                [p for p in predictions if p.get("expected_value") and p["expected_value"] > 1.0],
                key=lambda x: x["expected_value"],
                reverse=True,
            )

            if by_ev:
                # ケリー基準（分数ケリー: 0.25）で配分
                for p in by_ev[:3]:
                    prob = p["probability"]
                    odds = p["odds"]
                    if odds > 0 and prob > 0:
                        kelly = (prob * odds - 1) / (odds - 1) if odds > 1 else 0
                        kelly_fraction = max(0, kelly * 0.25)
                        amount = int(budget * kelly_fraction)
                        if amount >= 100:
                            bets.append({
                                "bet_type": "単勝",
                                "combination": str(p["car_number"]),
                                "amount": amount,
                                "reason": f"EV={p['expected_value']:.2f} {p['player_name']} (Kelly={kelly:.2%})",
                            })

            # 穴狙い
            long_shots = [
                p for p in predictions
                if p.get("odds", 0) >= 10 and p["probability"] >= 0.15
            ]
            if long_shots:
                ls = long_shots[0]
                bets.append({
                    "bet_type": "単勝",
                    "combination": str(ls["car_number"]),
                    "amount": budget // 5,
                    "reason": f"穴 {ls['player_name']} (オッズ{ls['odds']:.1f}倍, 確率{ls['probability']:.1%})",
                })

            # EVベースの買い目が無い場合は的中率ベースにフォールバック
            if not bets:
                bets.append({
                    "bet_type": "2車単",
                    "combination": f"{top3[0]['car_number']}-{top3[1]['car_number']}",
                    "amount": budget // 3,
                    "reason": f"EV不明のため的中率ベース: {top3[0]['player_name']} → {top3[1]['player_name']}",
                })

        return bets


def _analyze_comment_for_stats(comment: str | None) -> float:
    """統計予測用のコメント分析"""
    if not comment:
        return 0.0
    from .features import _analyze_comment
    return _analyze_comment(comment)
