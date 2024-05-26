from src.feature_selector import BaseFeatureSelector
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
import numpy as np


class Boruta(BaseFeatureSelector):
    """Boruta based feature selector."""

    def __init__(
        self,
        additional_feat_selector=None,
        model_n_estimators=100,
        model_max_depth=5,
        boruta_n_estimators="auto",
        boruta_max_iter=10,
    ) -> None:
        super().__init__()

        self.model = RandomForestClassifier(
            n_estimators=model_n_estimators, max_depth=model_max_depth
        )
        self.boruta_feat_selector = BorutaPy(
            verbose=2,
            estimator=self.model,
            n_estimators=boruta_n_estimators,
            max_iter=boruta_max_iter,
        )
        self.additional_feat_selector = additional_feat_selector

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.boruta_feat_selector.fit(X, y)

        if self.additional_feat_selector:
            X_reduced = self.boruta_feat_selector.transform(X)
            self.additional_feat_selector.fit(X_reduced, y)

    def get_support(self, indices: bool = True) -> np.ndarray:
        if indices:
            if self.additional_feat_selector:
                return self.additional_feat_selector.get_support(indices=True)
            else:
                return self.boruta_feat_selector.get_support(indices=True)

        if self.additional_feat_selector:
            return self.additional_feat_selector.get_support()
        else:
            return self.boruta_feat_selector.get_support()
