"""PredictMed API. Run: uvicorn app.main:app --reload (from the backend folder)."""
import os
from contextlib import asynccontextmanager
from types import SimpleNamespace
from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from . import drift, llm_router, pipeline, store, templates
from .model import RiskModel
from .rag import KB

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ctx = SimpleNamespace(model=None, kb=None)


def _log_mlflow(metrics, version):
    uri = os.getenv("MLFLOW_TRACKING_URI")
    if not uri:
        return "MLflow not configured"
    try:
        import mlflow
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment("predictmed")
        with mlflow.start_run(run_name=f"risk-model-v{version}") as r:
            mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, float)})
            mlflow.log_param("model_kind", metrics["kind"])
            return f"logged run {r.info.run_id}"
    except Exception as e:
        return f"MLflow error: {e}"


def _train():
    m = ctx.model or RiskModel()
    metrics = m.fit(pipeline.training_frame())
    ctx.model = m
    return metrics, _log_mlflow(metrics, m.version)


@asynccontextmanager
async def lifespan(app):
    store.init()
    if not store.members():
        pipeline.seed_demo()
    ctx.kb = KB(os.path.join(ROOT, "kb"))
    _train()
    if store.kv_get("baseline") is None:
        store.kv_set("baseline", [15, 30, 25, 20, 10])
    yield


app = FastAPI(title="PredictMed API", lifespan=lifespan)
api = APIRouter(prefix="/api")


class MemberIn(BaseModel):
    name: str
    age: int
    sex: str = "U"
    uacr: float
    nsaid: bool = False
    days_since_lab: int
    egfr: list[float] = Field(min_length=5, max_length=24)


class CsvIn(BaseModel):
    csv: str


class ApproveIn(BaseModel):
    lang: str = "en"
    text: str
    actor: str = "care-manager"


class RouteIn(BaseModel):
    text: str


def _find(mid):
    m = store.member(mid)
    if not m:
        raise HTTPException(404, "member not found")
    return pipeline.score(m, ctx.model)


@api.get("/health")
def health():
    return dict(ok=True, model=ctx.model.kind, model_version=ctx.model.version, rag=ctx.kb.mode,
                groq=bool(os.getenv("GROQ_API_KEY")), local_llm=os.getenv("HF_MODEL") or None,
                mlflow=os.getenv("MLFLOW_TRACKING_URI") or None, metrics=ctx.model.metrics)


@api.get("/members")
def members():
    return [{k: p[k] for k in ("id", "name", "age", "last", "stage", "dvi", "risk", "tier")}
            for p in pipeline.score_all(ctx.model)]


@api.get("/members/{mid}")
def member(mid: int):
    p = _find(mid)
    p["explanation"] = pipeline.explain(p, ctx.kb)
    return p


@api.post("/members")
def add(m: MemberIn):
    ok = store.add_member(m.model_dump())
    if not ok:
        raise HTTPException(409, "a member with that name already exists")
    return dict(ok=True)


@api.post("/members/import")
def import_csv(body: CsvIn):
    """CSV columns: name,age,uacr,nsaid,days_since_lab,e1..e8 (see README)."""
    ok = bad = 0
    for i, line in enumerate(body.csv.strip().splitlines()):
        f = [s.strip() for s in line.split(",")]
        try:
            if i == 0 and not f[1].replace(".", "").isdigit():
                continue
            n = [float(v) for v in f[1:]]
            if len(f) != 13 or not f[0]:
                raise ValueError
            added = store.add_member(dict(name=f[0], age=int(n[0]), uacr=n[1], nsaid=int(n[2]),
                                          days_since_lab=int(n[3]), egfr=n[4:]))
            ok, bad = ok + added, bad + (not added)
        except (ValueError, IndexError):
            bad += 1
    return dict(added=ok, rejected=bad)


@api.post("/members/{mid}/draft")
def draft(mid: int, lang: str = "en"):
    if lang not in templates.MSG:
        raise HTTPException(400, "lang must be en, te or hi")
    p = _find(mid)
    first = p["name"].split()[0]
    text, route = templates.draft(lang, first, p["tier"]), "template"
    if os.getenv("DRAFT_WITH_LLM") == "1":  # contains a name, so the router sends it to the LOCAL model only
        out = llm_router.generate("Rewrite this outreach message in the same language, warm and brief. Keep the meaning.",
                                  text, names=[first])
        if out["text"]:
            text, route = out["text"], out["route"]
    return dict(text=text, route=route, tier=p["tier"])


@api.post("/members/{mid}/approve")
def approve(mid: int, body: ApproveIn):
    p = _find(mid)
    if p["tier"] == "Low":
        raise HTTPException(400, "low-risk members do not get outreach")
    store.audit(body.actor, "approve_outreach", p["name"], f"{body.lang}: {body.text}")
    return dict(ok=True)


@api.get("/audit")
def audit():
    return store.audit_log()


@api.get("/drift")
def drift_status():
    cur = drift.dist([p["last"] for p in pipeline.score_all(ctx.model)])
    base = store.kv_get("baseline")
    p = drift.psi(cur, base)
    return dict(bins=[b[0] for b in drift.BINS], current=cur, baseline=base, psi=p,
                status=drift.status(p), model_version=ctx.model.version)


@api.post("/retrain")
def retrain():
    metrics, ml = _train()
    store.kv_set("baseline", drift.dist([p["last"] for p in pipeline.score_all(ctx.model)]))
    store.audit("system", "retrain", "-", f"v{ctx.model.version}; {ml}")
    return dict(model_version=ctx.model.version, metrics=metrics, mlflow=ml)


@api.post("/route")
def route(body: RouteIn):
    clean, n = llm_router.scrub(body.text, [m["name"] for m in store.members()])
    return dict(phi_found=n, route="local-hf" if n else "groq", deidentified_copy=clean)


app.include_router(api)
app.mount("/", StaticFiles(directory=os.path.join(os.path.dirname(ROOT), "frontend"), html=True), name="ui")
