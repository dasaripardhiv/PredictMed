"""Trajectory analytics: CKD-EPI 2021 eGFR, OLS slope, CUSUM change-point, DVI."""
import numpy as np


def ckd_epi_2021(scr_mg_dl: float, age: float, female: bool) -> float:
    """Race-free CKD-EPI 2021 creatinine equation (mL/min/1.73 m2)."""
    k = 0.7 if female else 0.9
    a = -0.241 if female else -0.302
    r = scr_mg_dl / k
    egfr = 142 * (min(r, 1) ** a) * (max(r, 1) ** -1.200) * (0.9938 ** age)
    return egfr * 1.012 if female else egfr


def ols(t, e):
    t, e = np.asarray(t, float), np.asarray(e, float)
    tm, em = t.mean(), e.mean()
    s = ((t - tm) * (e - em)).sum() / ((t - tm) ** 2).sum()
    return float(s), float(em - s * tm)


def cusum_alarm(t, e, baseline_n=4, k=1.0, h=6.0) -> int:
    """Index of first alarm, or -1. Fits a baseline on the first readings, then
    accumulates how far later readings fall below the extrapolated baseline."""
    if len(e) <= baseline_n:
        return -1
    b, c = ols(t[:baseline_n], e[:baseline_n])
    s = 0.0
    for i in range(baseline_n, len(e)):
        s = max(0.0, s + (c + b * t[i] - e[i]) - k)
        if s > h:
            return i
    return -1


def kdigo_stage(egfr: float) -> str:
    for lim, g in [(90, "G1"), (60, "G2"), (45, "G3a"), (30, "G3b"), (15, "G4")]:
        if egfr >= lim:
            return g
    return "G5"


def albuminuria(uacr: float) -> str:
    return "A3" if uacr > 300 else "A2" if uacr >= 30 else "A1"


def analyse(egfr, months):
    """Return slope, decline per year, CUSUM alarm index and the DVI (0-100).
    DVI here = 5 x decline per year + 20 if CUSUM alarms, capped at 100. Tune with clinicians."""
    t = [m / 12 for m in months]
    slope, icpt = ols(t, egfr)
    decline = max(0.0, -slope)
    al = cusum_alarm(t, egfr)
    dvi = min(100, round(5 * decline + (20 if al >= 0 else 0)))
    return dict(slope=slope, decline=decline, alarm_idx=al, dvi=dvi, last=float(egfr[-1]),
                fit=[icpt + slope * x for x in t])
