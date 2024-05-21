"""Module for feature selector classes."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseFeatureSelector(ABC):
    """Abstract class for feature selectors."""

    def __init__(self) -> None:
        """Initialize the feature selector instance."""
        pass

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the feature selector model to the data. This method should be implemented by the child class."""
        pass

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform the data using the feature selector model. This method should be implemented by the child class."""
        pass


class FeatureSelectorWrapper(BaseFeatureSelector):
    """Abstract class for feature selectors. Wraps already implemented feature selector model to unify the interface."""

    def __init__(self, model: Any) -> None:
        """Initialize the feature selector instance with a feature selector model."""
        super().__init__()
        self.model = model

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the feature selector model to the data."""
        self.model.fit(X, y)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform the data using the feature selector model."""
        return self.model.transform(X)
