"""Fixed binary-classification baselines for Phase 1.

Hyperparameters are shared across datasets. Phase 1 does not tune them.
"""

from sklearn.base import ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

MODEL_NAMES = ("logistic_regression", "random_forest", "gradient_boosting")


def get_models(random_state: int = 42) -> dict[str, ClassifierMixin]:
    """Return unfitted baseline classifiers keyed by model name.

    The same hyperparameters are used for every dataset. ``random_state`` is
    passed to every estimator that accepts it.
    """
    return {
        "logistic_regression": LogisticRegression(
            C=1.0,
            max_iter=5000,
            random_state=random_state,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=random_state,
        ),
    }
