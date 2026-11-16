"""Glue: member row -> trajectory -> features -> calibrated risk -> explanation."""
import os
from . import trajectory as tj, store, drift, llm_router
from .synth import features_from, cohort, DEMO, FEATURES
from .templates import reason as reason_tpl

TIERS = (0.45, 0.20)  # High >= 0.45, Medium >= 0.20. Agree thresholds with clinicians.
LABELS = {"decline": "eGFR decline speed", "alarm": "CUSUM alarm", "last_egfr": "Current eGFR",
          "uacr": "Albuminuria (UACR)", "nsaid": "NSAID flag", "days_since_lab": "Days since lab", "age": "Age"}


def seed_demo():
    for n, age, sex, u, x, d, e in DEMO:
        store.add_member(dict(name=n, age=age, sex=sex, uacr=u, nsaid=x, days_since_lab=d, egfr=e))


def training_frame():
    """HOOK: replace with real labelled outcomes (features + a 'label' column your clinical
    team defines, e.g. rapid progression in the following 12 months). Keep time order."""
    return cohort()


def score(m, model):
    months = [3 * i for i in range(len(m["egfr"]))]
    a = tj.analyse(m["egfr"], months)
    x = features_from(a, m["uacr"], m["nsaid"], m["days_since_lab"], m["age"])
    risk = float(model.predict([x])[0])
    c = model.contributions(x)
    return dict(id=m["id"], name=m["name"], age=m["age"], series=m["egfr"], months=months, fit=a["fit"],
                alarm_idx=a["alarm_idx"], decline=a["decline"], slope=a["slope"], dvi=a["dvi"], last=a["last"],
                stage=tj.kdigo_stage(a["last"]), albuminuria=tj.albuminuria(m["uacr"]), uacr=m["uacr"],
                nsaid=bool(m["nsaid"]), days_since_lab=m["days_since_lab"], risk=risk,
                tier="High" if risk >= TIERS[0] else "Medium" if risk >= TIERS[1] else "Low",
                contributions=[dict(feature=LABELS[k], value=v) for k, v in sorted(c.items(), key=lambda kv: -abs(kv[1]))])


def score_all(model):
    return sorted((score(m, model) for m in store.members()), key=lambda s: -s["risk"])


def explain(p, kb):
    """Facts contain NO name, so this text may go to Groq. KDIGO excerpts come from RAG."""
    facts = (f"eGFR fell from {p['series'][0]:.0f} to {p['series'][-1]:.0f} over {p['months'][-1]} months "
             f"({p['decline']:.1f} per year). KDIGO stage {p['stage']}, albuminuria {p['albuminuria']}. "
             f"CUSUM alarm: {'yes' if p['alarm_idx'] >= 0 else 'no'}. NSAID on record: {'yes' if p['nsaid'] else 'no'}. "
             f"Days since last lab: {p['days_since_lab']}.")
    hits = kb.search(f"{p['stage']} {p['albuminuria']} eGFR decline progression referral monitoring NSAID", 3)
    cites = [dict(n=i + 1, source=s, text=t) for i, (s, t) in enumerate(hits)]
    ctx = "\n".join(f"[{c['n']}] {c['text']}" for c in cites)
    system = ("You support care managers. Use ONLY the facts and excerpts given. Write 2-3 plain sentences "
              "explaining why this member is flagged. Cite excerpts like [1]. Do not give treatment advice.")
    out = llm_router.generate(system, f"Facts: {facts}\n\nExcerpts:\n{ctx}", names=[m["name"] for m in store.members()])
    if out["text"]:
        return dict(text=out["text"], route=out["route"], mode="llm", citations=cites)
    return dict(text=reason_tpl("This member", p), route="template", mode="template", citations=cites)
