"""
random_forest.py
The MIT License (MIT)
Copyright (c) 2026 Science Solutions International Laboratory, Inc.
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from core.optimizer.surrogate_models.protocol import SurrogateProtocol


class RandomForestSurrogate(SurrogateProtocol):
    """Random forest surrogate model with built-in feature importances."""

    def __init__(
        self,
        *,
        n_estimators: int = 200,
        max_depth: int | None = None,
        random_state: int | None = 0,
    ) -> None:
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
        )
        self._has_fit = False

    def is_ready(self) -> bool:
        return self._has_fit

    def fit(self, x: np.ndarray, y: np.ndarray, **kw: object) -> None:
        _ = kw
        self.model.fit(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        self._has_fit = True

    def update(self, x: np.ndarray, y: np.ndarray) -> None:
        self.fit(x, y)

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        if not self.is_ready():
            msg = "RandomForestSurrogate must be fitted before prediction."
            raise RuntimeError(msg)
        mu = np.asarray(self.model.predict(np.asarray(x, dtype=float)), dtype=float)
        if mu.ndim == 1:
            mu = mu.reshape(-1, 1)
        sigma = np.zeros_like(mu)
        return mu, sigma

    def get_feature_importances(self) -> np.ndarray:
        if not self.is_ready():
            msg = "RandomForestSurrogate must be fitted before feature importances are available."
            raise RuntimeError(msg)
        return np.asarray(self.model.feature_importances_, dtype=float)
