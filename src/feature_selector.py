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
