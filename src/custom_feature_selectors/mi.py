"""Module for Mutual Information based feature selection."""

from typing import Literal, Union
import numpy as np
from sklearn.metrics import mutual_info_score as MI

from src.feature_selector import BaseFeatureSelector
from src.custom_feature_selectors.mi_utils import CMI, check_for_stopping_rule


class CMIM(BaseFeatureSelector):
    """Mutual Information based feature selection."""

    def __init__(self, n_features: Union[int, Literal["auto"]] = "auto"):
        """Initialize the CMIM feature selector.

        Arguments:
            n_features: Number of features to select. If "auto", the stopping rule is used.
        """
        self.n_features = n_features
        self._selected = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the CMIM feature selector to the data.

        Arguments:
            X: Array with training features.
            y: Array with training target variable.
        """
        selected = []
        for _ in range(X.shape[1] if self.n_features == "auto" else self.n_features):
            max_cmim_value = float("-inf")
            for i in range(X.shape[1]):
                if i in selected:
                    continue
                J = MI(X[:, i], y)
                max_value = float("-inf")
                for j in selected:
                    curr_value = MI(X[:, i], X[:, j]) - CMI(X[:, i], X[:, j], y)
                    if curr_value > max_value:
                        max_value = curr_value
                if J - max_value > max_cmim_value:
                    max_cmim_value = J - max_value
                    max_idx = i

            if self.n_features == "auto" and check_for_stopping_rule(
                max_idx, X, y, selected
            ):
                break
            selected.append(max_idx)
        selected.sort()
        self._selected = selected

    def get_support(self, indices: bool = True) -> np.ndarray:
        """
        Get indices of the chosen features after the fit.

        Arguments:
            indices: If True, the return value will be an array of integers, rather than a boolean mask.
        """
        if indices:
            return np.array(self._selected)
        mask = np.zeros(len(self._selected), dtype=bool)
        mask[self._selected] = True
        return mask
