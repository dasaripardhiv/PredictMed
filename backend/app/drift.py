"""Population Stability Index on the latest-eGFR mix."""
import math

BINS = [("90+", 90, 1e9), ("60-89", 60, 90), ("45-59", 45, 60), ("30-44", 30, 45), ("under 30", -1e9, 30)]


def dist(vals):
    n = len(vals) or 1
    return [100 * sum(lo <= v < hi for v in vals) / n for _, lo, hi in BINS]


def psi(cur, base):
    s = 0.0
    for c, b in zip(cur, base):
        a, e = max(c, .5) / 100, max(b, .5) / 100
        s += (a - e) * math.log(a / e)
    return s


def status(p):
    return "stable" if p < .1 else "watch" if p < .25 else "drift"
