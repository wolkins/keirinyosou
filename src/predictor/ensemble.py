"""アンサンブル: LambdaRank + Binary分類スタッキング

Layer 1:
  - LightGBM LambdaRank (順位スコア)
  - LightGBM Binary (1着判定)
  - LightGBM Binary (3着内判定)
Layer 2:
  - Ridge回帰メタモデルで統合
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except (ImportError, OSError):
    HAS_LIGHTGBM = False

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.common.config import PROJECT_ROOT

MODEL_DIR = PROJECT_ROOT / "data" / "models"


class EnsemblePredictor:
    """スタッキングアンサンブル"""

    def __init__(self):
        self.ranker = None        # LambdaRank
        self.win_clf = None       # 1着判定
        self.top3_clf = None      # 3着内判定
        self.meta_model = None    # Layer2メタモデル
        self.scaler = None
        self.feature_cols = []
        self._model_path = MODEL_DIR / "ensemble.pkl"

    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              group_train: list, w_train: pd.Series,
              X_valid: pd.DataFrame, y_valid: pd.Series,
              group_valid: list,
              finish_positions_train: pd.Series,
              finish_positions_valid: pd.Series) -> dict:
        """Layer1 + Layer2 を学習"""
        if not HAS_LIGHTGBM:
            return {"error": "LightGBM is required for ensemble"}

        self.feature_cols = list(X_train.columns)

        # === Layer 1: 3つのモデルを学習 ===

        # 1. LambdaRank
        self.ranker = lgb.LGBMRanker(
            objective="lambdarank", metric="ndcg", ndcg_eval_at=[1, 3],
            learning_rate=0.03, num_leaves=31,
            min_data_in_leaf=max(30, len(X_train) // 80),
            feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=5,
            lambda_l1=0.1, lambda_l2=1.0, n_estimators=1500, verbose=-1,
        )
        callbacks = [lgb.early_stopping(50, verbose=False)]
        if len(X_valid) > 0 and group_valid:
            self.ranker.fit(
                X_train, y_train, group=group_train, sample_weight=w_train,
                eval_set=[(X_valid, y_valid)], eval_group=[group_valid],
                callbacks=callbacks,
            )
        else:
            self.ranker.fit(X_train, y_train, group=group_train, sample_weight=w_train)

        # 2. Binary: 1着判定
        y_win_train = (finish_positions_train == 1).astype(int)
        y_win_valid = (finish_positions_valid == 1).astype(int)

        self.win_clf = lgb.LGBMClassifier(
            objective="binary", metric="binary_logloss",
            learning_rate=0.03, num_leaves=31,
            min_data_in_leaf=max(30, len(X_train) // 80),
            feature_fraction=0.7, n_estimators=1000, verbose=-1,
            scale_pos_weight=len(y_win_train) / max(1, y_win_train.sum()) / 2,
        )
        if len(X_valid) > 0:
            self.win_clf.fit(
                X_train, y_win_train, sample_weight=w_train,
                eval_set=[(X_valid, y_win_valid)],
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )
        else:
            self.win_clf.fit(X_train, y_win_train, sample_weight=w_train)

        # 3. Binary: 3着内判定
        y_top3_train = (finish_positions_train <= 3).astype(int)
        y_top3_valid = (finish_positions_valid <= 3).astype(int)

        self.top3_clf = lgb.LGBMClassifier(
            objective="binary", metric="binary_logloss",
            learning_rate=0.03, num_leaves=31,
            min_data_in_leaf=max(30, len(X_train) // 80),
            feature_fraction=0.7, n_estimators=1000, verbose=-1,
        )
        if len(X_valid) > 0:
            self.top3_clf.fit(
                X_train, y_top3_train, sample_weight=w_train,
                eval_set=[(X_valid, y_top3_valid)],
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )
        else:
            self.top3_clf.fit(X_train, y_top3_train, sample_weight=w_train)

        # === Layer 2: メタモデル ===
        # 検証データでLayer1の予測を生成
        meta_train = self._get_meta_features(X_valid)
        meta_target = y_valid  # ランクラベル

        self.scaler = StandardScaler()
        meta_train_scaled = self.scaler.fit_transform(meta_train)

        self.meta_model = Ridge(alpha=1.0)
        self.meta_model.fit(meta_train_scaled, meta_target)

        # 保存
        self.save()

        return {
            "layer1_models": ["LambdaRank", "WinBinary", "Top3Binary"],
            "layer2_model": "Ridge",
            "meta_features": meta_train.shape[1],
        }

    def _get_meta_features(self, X: pd.DataFrame) -> np.ndarray:
        """Layer1の出力をスタック"""
        rank_scores = self.ranker.predict(X)
        win_proba = self.win_clf.predict_proba(X)[:, 1]
        top3_proba = self.top3_clf.predict_proba(X)[:, 1]

        return np.column_stack([rank_scores, win_proba, top3_proba])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """アンサンブル予測スコアを返す"""
        if self.meta_model is None:
            # メタモデルがなければRankerのみ
            return self.ranker.predict(X)

        meta = self._get_meta_features(X)
        meta_scaled = self.scaler.transform(meta)
        return self.meta_model.predict(meta_scaled)

    def predict_detail(self, X: pd.DataFrame) -> dict:
        """各Layer1モデルの予測も含めて返す"""
        rank_scores = self.ranker.predict(X)
        win_proba = self.win_clf.predict_proba(X)[:, 1]
        top3_proba = self.top3_clf.predict_proba(X)[:, 1]

        if self.meta_model is not None:
            meta = np.column_stack([rank_scores, win_proba, top3_proba])
            meta_scaled = self.scaler.transform(meta)
            ensemble_scores = self.meta_model.predict(meta_scaled)
        else:
            ensemble_scores = rank_scores

        return {
            "ensemble_score": ensemble_scores,
            "rank_score": rank_scores,
            "win_proba": win_proba,
            "top3_proba": top3_proba,
        }

    def save(self):
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        with open(self._model_path, "wb") as f:
            pickle.dump({
                "ranker": self.ranker,
                "win_clf": self.win_clf,
                "top3_clf": self.top3_clf,
                "meta_model": self.meta_model,
                "scaler": self.scaler,
                "feature_cols": self.feature_cols,
            }, f)

    def load(self) -> bool:
        if self._model_path.exists():
            with open(self._model_path, "rb") as f:
                data = pickle.load(f)
            self.ranker = data["ranker"]
            self.win_clf = data["win_clf"]
            self.top3_clf = data["top3_clf"]
            self.meta_model = data["meta_model"]
            self.scaler = data["scaler"]
            self.feature_cols = data.get("feature_cols", [])
            return True
        return False
