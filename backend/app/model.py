"""LightGBM risk model + isotonic calibration. Falls back to logistic regression if
LightGBM is not installed so the app still runs."""
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from .synth import FEATURES

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover
    lgb = None


class RiskModel:
    def __init__(self):
        self.base = self.iso = self.mean = None
        self.kind, self.version, self.metrics = "untrained", 0, {}

    def fit(self, df):
        """Time-based split: oldest 60% train, next 20% calibrate, newest 20% test."""
        n = len(df); a, b = int(n * .6), int(n * .8)
        X, y = df[FEATURES].to_numpy(float), df["label"].to_numpy(int)
        if lgb:
            self.base = lgb.LGBMClassifier(n_estimators=150, learning_rate=.05, num_leaves=15,
                                           min_child_samples=30, verbose=-1)
            self.kind = "lightgbm"
        else:
            self.base, self.kind = LogisticRegression(max_iter=3000), "logistic (LightGBM not installed)"
        self.base.fit(X[:a], y[:a])
        self.mean = X[:a].mean(0)
        self.iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(self._raw(X[a:b]), y[a:b])
        p = self._raw(X[b:])
        self.version += 1
        self.metrics = dict(kind=self.kind, auc=float(roc_auc_score(y[b:], p)),
                            brier_raw=float(brier_score_loss(y[b:], p)),
                            brier_calibrated=float(brier_score_loss(y[b:], self.iso.predict(p))),
                            n_train=a, n_calib=b - a, n_test=n - b)
        return self.metrics

    def _raw(self, X):
        return self.base.predict_proba(np.asarray(X, float))[:, 1]

    def predict(self, X):
        return self.iso.predict(self._raw(X))

    def contributions(self, x):
        """Per-feature contribution to the raw log-odds. LightGBM's pred_contrib is TreeSHAP."""
        x = np.asarray(x, float).reshape(1, -1)
        c = self.base.booster_.predict(x, pred_contrib=True)[0][:-1] if self.kind == "lightgbm" \
            else self.base.coef_[0] * (x[0] - self.mean)
        return {f: float(v) for f, v in zip(FEATURES, c)}
