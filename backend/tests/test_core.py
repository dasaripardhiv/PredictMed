"""Run: python tests/test_core.py   (or pytest) from the backend folder."""
import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["DB_PATH"] = os.path.join(tempfile.mkdtemp(), "t.db")
from app import trajectory as tj, drift, llm_router, store, pipeline
from app.model import RiskModel
from app.synth import cohort, DEMO, MONTHS
from app.rag import KB


def test_ckd_epi():
    assert abs(tj.ckd_epi_2021(1.0, 60, False) - 86.2) < 1.5
    assert tj.ckd_epi_2021(1.0, 60, True) > 0


def test_cusum():
    t = [m / 12 for m in MONTHS]
    assert tj.cusum_alarm(t, DEMO[0][6]) == 5      # rapid decliner
    assert tj.cusum_alarm(t, DEMO[2][6]) == -1     # stable


def test_stage_and_albuminuria():
    assert tj.kdigo_stage(41) == "G3b" and tj.kdigo_stage(90) == "G1" and tj.albuminuria(320) == "A3"


def test_psi():
    assert drift.psi([15, 30, 25, 20, 10], [15, 30, 25, 20, 10]) < 1e-9
    assert drift.psi([0, 40, 10, 40, 10], [15, 30, 25, 20, 10]) > .25


def test_router():
    clean, n = llm_router.scrub("Ravi K. phone 9876543210 DOB 12/03/1967 a@b.com", ["Ravi K."])
    assert n == 4 and "Ravi" not in clean and "9876543210" not in clean
    assert llm_router.scrub("eGFR is falling", ["Ravi K."])[1] == 0
    os.environ.pop("GROQ_API_KEY", None); os.environ.pop("HF_MODEL", None)
    out = llm_router.generate("s", "Ravi has 9876543210", ["Ravi K."])
    assert out["route"] == "local-hf" and out["text"] is None   # PHI never falls back to Groq


def test_model_and_pipeline():
    m = RiskModel(); met = m.fit(cohort())
    assert met["auc"] > .75 and met["brier_calibrated"] <= met["brier_raw"] + .02
    store.init(); pipeline.seed_demo()
    ranked = pipeline.score_all(m)
    assert len(ranked) == 8 and ranked[0]["risk"] >= ranked[-1]["risk"]
    assert ranked[-1]["tier"] == "Low" and ranked[0]["tier"] == "High"
    kb = KB(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "kb"))
    e = pipeline.explain(ranked[0], kb)
    assert e["mode"] in ("template", "llm") and len(e["citations"]) == 3
    print("model:", met["kind"], "auc %.2f" % met["auc"], "| rag:", kb.mode, "| ranked:", [(r["name"], r["tier"], round(r["risk"], 2)) for r in ranked])


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f(); print("ok", k)
