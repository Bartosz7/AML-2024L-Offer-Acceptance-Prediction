from pydantic import BaseModel
from functools import cached_property
from sklearn.base import ClassifierMixin
from typing import Dict, Any, Callable, Optional, List
import uuid


class Experiment(BaseModel):
    classifier: Callable
    classifier_config: Dict[str, Any] = {}
    feature_selector: Callable
    feature_selector_config: Dict[str, Any] = {}
    scores: Optional[List[int]] = None

    @cached_property
    def experiment_name(self):
        return (
            "exp_"
            + "".join([c for c in self.classifier.__name__ if c.isupper()])
            + "_"
            + "".join([c for c in self.feature_selector.__name__ if c.isupper()])
            + "_"
            + str(uuid.uuid4().hex[:6])
        ).lower()

    class Config:
        arbitrary_types_allowed = True
