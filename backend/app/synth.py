"""Synthetic data. Replace with real data (see README: Data connection)."""
import numpy as np
import pandas as pd
from .trajectory import analyse

MONTHS = [0, 3, 6, 9, 12, 15, 18, 21]
FEATURES = ["decline", "alarm", "last_egfr", "uacr", "nsaid", "days_since_lab", "age"]

DEMO = [  # name, age, sex, uacr, nsaid, days_since_lab, 8 eGFR readings
    ("Ravi K.", 58, "M", 320, 1, 40, [68, 67, 66, 63, 58, 52, 47, 41]),
    ("Lakshmi P.", 64, "F", 85, 0, 60, [55, 54, 55, 53, 52, 51, 49, 48]),
    ("Anil S.", 49, "M", 12, 0, 30, [92, 91, 92, 90, 91, 90, 89, 90]),
    ("Sunita R.", 71, "F", 410, 0, 95, [38, 37, 36, 35, 33, 31, 28, 26]),
    ("Mohan T.", 53, "M", 45, 1, 50, [74, 73, 72, 70, 66, 63, 60, 57]),
    ("Fatima B.", 60, "F", 22, 0, 45, [81, 80, 80, 79, 78, 77, 77, 76]),
    ("Karthik V.", 45, "M", 150, 0, 35, [88, 87, 85, 82, 78, 73, 69, 64]),
    ("Padma N.", 67, "F", 60, 0, 150, [47, 47, 46, 46, 45, 44, 44, 43]),
]


def features_from(a, uacr, nsaid, days, age):
    return [a["decline"], 1 if a["alarm_idx"] >= 0 else 0, a["last"], uacr, nsaid, days, age]


def cohort(n=4000, seed=7) -> pd.DataFrame:
    """Time-ordered synthetic cohort. Labels come from a hidden rule plus noise, so
    model metrics prove the plumbing works, NOT clinical validity."""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        start, rate = rng.uniform(25, 100), rng.gamma(2, 1.6)
        t = np.array(MONTHS) / 12
        extra = np.where(np.arange(8) >= 4, rng.uniform(0, 3) * rng.uniform(0, 4) * (t - t[3]), 0) if rng.random() < .3 else 0
        e = np.clip(start - rate * t - extra + rng.normal(0, 1.2, 8), 5, 120)
        uacr, nsaid = float(rng.lognormal(3.8, 1.3)), int(rng.random() < .2)
        days, age = int(rng.integers(10, 180)), float(np.clip(rng.normal(60, 12), 30, 85))
        a = analyse(list(e), MONTHS)
        f = features_from(a, uacr, nsaid, days, age)
        z = 1.7 * (-3.0 + .13 * min(a["decline"], 15) + .9 * f[1] + (.9 if uacr > 300 else .5 if uacr >= 30 else 0)
             + (.03 * (60 - a["last"]) if a["last"] < 60 else 0) + .3 * nsaid + .004 * days) + .2
        rows.append(f + [int(rng.random() < 1 / (1 + np.exp(-z)))])
    return pd.DataFrame(rows, columns=FEATURES + ["label"])
