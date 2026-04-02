"""確率キャリブレーション: LambdaRankスコア → 校正確率"""
import pickle
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression

from src.common.config import PROJECT_ROOT

MODEL_DIR = PROJECT_ROOT / "data" / "models"


class ProbabilityCalibrator:
    """Isotonic回帰による確率キャリブレーション

    LambdaRankの生スコアを校正された確率に変換する。
    """

    def __init__(self, name: str = "win"):
        """
        Args:
            name: キャリブレータ名 ("win" or "top3")
        """
        self.name = name
        self.model = IsotonicRegression(out_of_bounds="clip")
        self._fitted = False
        self._save_path = MODEL_DIR / f"calibrator_{name}.pkl"

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> "ProbabilityCalibrator":
        """キャリブレータを学習

        Args:
            scores: LambdaRankの生スコア
            labels: binaryラベル (win: 1着=1, それ以外=0 / top3: 3着以内=1, それ以外=0)

        Returns:
            self
        """
        self.model.fit(scores, labels)
        self._fitted = True
        return self

    def predict_proba(self, scores: np.ndarray) -> np.ndarray:
        """校正された確率を返す

        Args:
            scores: LambdaRankの生スコア

        Returns:
            校正された確率 (0-1)
        """
        if not self._fitted:
            # 未学習の場合はsigmoid近似でフォールバック
            return 1.0 / (1.0 + np.exp(-scores))
        return self.model.predict(scores)

    def save(self) -> Path:
        """キャリブレータをファイルに保存"""
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        with open(self._save_path, "wb") as f:
            pickle.dump({"model": self.model, "fitted": self._fitted, "name": self.name}, f)
        return self._save_path

    def load(self) -> bool:
        """キャリブレータをファイルから読み込み

        Returns:
            読み込み成功ならTrue
        """
        if self._save_path.exists():
            with open(self._save_path, "rb") as f:
                data = pickle.load(f)
            self.model = data["model"]
            self._fitted = data.get("fitted", True)
            self.name = data.get("name", self.name)
            return True
        return False

    @property
    def is_fitted(self) -> bool:
        return self._fitted
