"""Small, locally trained pair classifiers.

Model selection and decision thresholds belong to validation code. This module
only fits models and returns pair-level probabilities.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def fit_pair_models(features: np.ndarray, labels: np.ndarray) -> dict[str, object]:
    """Fit an interpretable linear baseline and a modest nonlinear matcher."""
    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.uint8)
    if features.ndim != 2 or len(features) != len(labels):
        raise ValueError("features and labels must have matching first dimension")
    if len(np.unique(labels)) != 2:
        raise ValueError("training pairs must include both classes")
    linear = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=500, random_state=17),
    )
    boosted = HistGradientBoostingClassifier(
        max_iter=150,
        max_leaf_nodes=15,
        learning_rate=0.08,
        l2_regularization=1.0,
        random_state=17,
    )
    linear.fit(features, labels)
    boosted.fit(features, labels)
    return {"logistic": linear, "hist_gradient_boosting": boosted}


def predict_probabilities(model: object, features: np.ndarray) -> np.ndarray:
    """Return positive-class probabilities for final blocked candidates."""
    return model.predict_proba(np.asarray(features, dtype=np.float32))[:, 1]
