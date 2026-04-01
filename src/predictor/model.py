"""予測モデル

2つのモード:
- accuracy (的中率重視): 1着を当てることを最適化
- roi (回収率重視): 期待値の高い買い目を探す
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sqlalchemy.orm import Session

from src.common.config import PROJECT_ROOT
from src.common.database import Odds, Race, RaceEntry, get_session

from .features import FEATURE_COLUMNS, build_features_for_race

MODEL_DIR = PROJECT_ROOT / "data" / "models"


class KeirinPredictor:
    """競輪予測モデル"""

    def __init__(self, mode: str = "accuracy"):
        """
        Args:
            mode: "accuracy" (的中率重視) or "roi" (回収率重視)
        """
        self.mode = mode
        self.model = None
        self._model_path = MODEL_DIR / f"model_{mode}.pkl"

    def train(self, session: Session, min_races: int = 50) -> dict:
        """過去のレース結果から学習

        Returns:
            {"n_races": int, "n_samples": int, "cv_score": float}
        """
        # 結果が確定しているレースを取得
        races = session.query(Race).filter(Race.status == "finished").all()
        if len(races) < min_races:
            return {"error": f"学習データが不足しています ({len(races)}/{min_races}レース)"}

        all_features = []
        for race in races:
            df = build_features_for_race(session, race)
            if not df.empty and "finish_position" in df.columns:
                all_features.append(df)

        if not all_features:
            return {"error": "有効なトレーニングデータがありません"}

        train_df = pd.concat(all_features, ignore_index=True)

        X = train_df[FEATURE_COLUMNS].fillna(0)
        y = train_df["finish_position"]

        if self.mode == "accuracy":
            # 3着以内かどうかの分類
            y_class = (y <= 3).astype(int)
            self.model = GradientBoostingClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
            )
            scores = cross_val_score(self.model, X, y_class, cv=5, scoring="accuracy")
            self.model.fit(X, y_class)
        else:
            # 着順の回帰（小さいほど良い）
            self.model = GradientBoostingRegressor(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
            )
            scores = cross_val_score(self.model, X, y, cv=5, scoring="neg_mean_absolute_error")
            self.model.fit(X, y)

        # モデル保存
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        with open(self._model_path, "wb") as f:
            pickle.dump(self.model, f)

        return {
            "n_races": len(races),
            "n_samples": len(train_df),
            "cv_score": float(np.mean(scores)),
            "cv_std": float(np.std(scores)),
        }

    def load(self) -> bool:
        """保存済みモデルを読み込み"""
        if self._model_path.exists():
            with open(self._model_path, "rb") as f:
                self.model = pickle.load(f)
            return True
        return False

    def predict(self, session: Session, race: Race) -> list[dict]:
        """レースの予想を行う

        Returns:
            [{"car_number": int, "player_name": str, "score": float,
              "probability": float, "recommendation": str}, ...]
            scoreが高いほど上位予想
        """
        if self.model is None:
            if not self.load():
                return self._predict_by_stats(session, race)

        df = build_features_for_race(session, race)
        if df.empty:
            return []

        X = df[FEATURE_COLUMNS].fillna(0)

        if self.mode == "accuracy":
            # 3着以内の確率
            probs = self.model.predict_proba(X)
            if probs.shape[1] >= 2:
                scores = probs[:, 1]  # 3着以内の確率
            else:
                scores = probs[:, 0]
        else:
            # 着順予測（小さいほど良い → 反転してスコアに）
            pred_positions = self.model.predict(X)
            scores = 1.0 / (pred_positions + 0.1)  # 着順が小さいほど高スコア

        # オッズ情報を取得して期待値を計算
        odds_data = session.query(Odds).filter_by(race_id=race.id, bet_type="win").all()
        odds_map = {}
        for o in odds_data:
            try:
                car_num = int(o.combination)
                odds_map[car_num] = o.odds_value
            except ValueError:
                pass

        # エントリー情報と結合
        entries = session.query(RaceEntry).filter_by(race_id=race.id).all()
        entry_map = {e.car_number: e for e in entries}

        results = []
        for i, row in df.iterrows():
            car_num = int(row["car_number"])
            entry = entry_map.get(car_num)
            score = float(scores[i])

            # 期待値計算（回収率モード用）
            odds_val = odds_map.get(car_num, 0)
            expected_value = score * odds_val if odds_val else score

            # 推奨度
            if self.mode == "roi":
                final_score = expected_value
            else:
                final_score = score

            results.append({
                "car_number": car_num,
                "player_name": entry.player.name if entry and entry.player else "不明",
                "grade": entry.player.grade if entry and entry.player else "",
                "win_rate": entry.win_rate or 0,
                "second_rate": entry.second_rate or 0 if entry else 0,
                "third_rate": entry.third_rate or 0 if entry else 0,
                "score": round(final_score, 4),
                "probability": round(score, 4),
                "odds": odds_val,
                "expected_value": round(expected_value, 4) if odds_val else None,
                "recommendation": self._get_recommendation(final_score),
                "comment": entry.comment or "" if entry else "",
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    def _predict_by_stats(self, session: Session, race: Race) -> list[dict]:
        """モデルが無い場合の統計ベース予測

        勝率・連対率・級班などから簡易スコアを算出。
        """
        entries = session.query(RaceEntry).filter_by(race_id=race.id).all()
        if not entries:
            return []

        results = []
        for entry in entries:
            # 簡易スコア = 勝率×3 + 2連対率×2 + 3連対率 + 級班ボーナス + 車番ボーナス
            win_r = entry.win_rate or 0
            sec_r = entry.second_rate or 0
            thi_r = entry.third_rate or 0

            grade_bonus = 0
            if entry.player and entry.player.grade:
                grade_map = {"SS": 10, "S1": 7, "S2": 5, "A1": 3, "A2": 1, "A3": 0}
                grade_bonus = grade_map.get(entry.player.grade, 0)

            car_bonus = max(0, (6 - entry.car_number)) * 0.5

            score = (win_r * 3 + sec_r * 2 + thi_r) / 6 + grade_bonus + car_bonus
            comment_score = 0
            if entry.comment:
                from .features import _analyze_comment
                comment_score = _analyze_comment(entry.comment) * 5

            score += comment_score

            # オッズ
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
                "odds": odds_val,
                "expected_value": round(score / 100 * odds_val, 4) if odds_val else None,
                "recommendation": self._get_recommendation(score / 100),
                "comment": entry.comment or "",
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    def _get_recommendation(self, score: float) -> str:
        """スコアから推奨度を判定"""
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
        """推奨買い目を生成

        Args:
            predictions: predict()の結果
            budget: 予算(円)

        Returns:
            [{"bet_type": str, "combination": str, "amount": int, "reason": str}, ...]
        """
        if not predictions:
            return []

        top3 = predictions[:3]
        bets = []

        if self.mode == "accuracy":
            # 的中率重視: 本命を厚く
            # 2車単: 1位→2位, 1位→3位
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
            # 2車複: 上位2名
            bets.append({
                "bet_type": "2車複",
                "combination": f"{top3[0]['car_number']}={top3[1]['car_number']}",
                "amount": budget // 4,
                "reason": f"上位2名の複勝: {top3[0]['player_name']}, {top3[1]['player_name']}",
            })

        else:
            # 回収率重視: 期待値が高い買い目を
            # 期待値でソート
            by_ev = sorted(
                [p for p in predictions if p.get("expected_value")],
                key=lambda x: x["expected_value"],
                reverse=True,
            )

            if by_ev:
                # 期待値1位の単勝
                bets.append({
                    "bet_type": "単勝",
                    "combination": str(by_ev[0]["car_number"]),
                    "amount": budget // 3,
                    "reason": f"期待値最大: {by_ev[0]['player_name']} (EV={by_ev[0]['expected_value']:.2f})",
                })

                # 高期待値の2車単
                if len(by_ev) >= 2:
                    bets.append({
                        "bet_type": "2車単",
                        "combination": f"{by_ev[0]['car_number']}-{by_ev[1]['car_number']}",
                        "amount": budget // 4,
                        "reason": f"高期待値: {by_ev[0]['player_name']} → {by_ev[1]['player_name']}",
                    })

                # 穴狙い: オッズが高く確率もそこそこある選手
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
                        "reason": f"穴狙い: {ls['player_name']} (オッズ{ls['odds']:.1f}倍, 確率{ls['probability']:.1%})",
                    })

        return bets
