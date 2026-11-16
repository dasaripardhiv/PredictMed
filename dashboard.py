
"""PredictMed Dashboard: visual control center.

Run:
    python -m streamlit run dashboard.py --server.port 8502

Demonstration prototype. Not clinically validated.
"""

import os
from datetime import datetime

import mlflow
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from mlflow.tracking import MlflowClient


# ============================================================
# CONFIGURATION
# ============================================================

API_URL = os.getenv(
    "PREDICTMED_API_URL",
    "http://127.0.0.1:8000",
).rstrip("/")

MLFLOW_URL = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5000",
)

mlflow.set_tracking_uri(MLFLOW_URL)

st.set_page_config(
    page_title="PredictMed Dashboard",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# STYLING
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background: #f5f8fb;
        color: #183044;
    }

    [data-testid="stHeader"] {
        background: rgba(245,248,251,.95);
    }

    .block-container {
        max-width: 1450px;
        padding-top: 1.2rem;
        padding-bottom: 2.5rem;
    }

    [data-testid="stSidebar"] {
        background: #15364a;
    }

    [data-testid="stSidebar"] * {
        color: #f4fbff;
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #d5e8f0;
    }

    h1, h2, h3 {
        color: #15364a;
        letter-spacing: -.02em;
    }

    .hero {
        padding: 25px 28px;
        border-radius: 18px;
        color: white;
        background: linear-gradient(115deg, #15364a, #087f8c);
        margin-bottom: 20px;
    }

    .hero h1 {
        color: white !important;
        margin: 0 0 8px 0;
        font-size: 2rem !important;
    }

    .hero p {
        color: #e0fbf8;
        margin: 0;
    }

    .eyebrow {
        font-size: .76rem;
        text-transform: uppercase;
        letter-spacing: .12em;
        font-weight: 700;
        color: #9ff4e7;
        margin-bottom: 8px;
    }

    .section-note {
        color: #637789;
        margin-top: -8px;
        margin-bottom: 12px;
    }

    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #e0e9ef;
        border-radius: 14px;
        padding: 12px 15px;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid #e0e9ef;
        border-radius: 12px;
        overflow: hidden;
    }

    div.stButton > button,
    div.stDownloadButton > button {
        border-radius: 9px;
        font-weight: 650;
    }

    .notice {
        background: #e8f6f4;
        border-left: 4px solid #087f8c;
        padding: 12px 15px;
        border-radius: 8px;
        color: #244c59;
        margin: 12px 0;
    }

    .footer {
        border-top: 1px solid #dce7ed;
        margin-top: 28px;
        padding-top: 14px;
        font-size: .8rem;
        color: #627686;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_api(path):
    """Return (JSON data, error message)."""
    try:
        response = requests.get(
            f"{API_URL}{path}",
            timeout=5,
        )
        response.raise_for_status()
        return response.json(), None
    except (requests.RequestException, ValueError) as exc:
        return None, str(exc)


def as_records(data):
    """Convert common API response formats into a list of dictionaries."""
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]

    if isinstance(data, dict):
        for key in (
            "members",
            "items",
            "results",
            "audit",
            "events",
            "data",
        ):
            value = data.get(key)
            if isinstance(value, list):
                return [
                    item for item in value
                    if isinstance(item, dict)
                ]

    return []


def first_value(row, *keys, default=None):
    """Return the first non-empty value from a dictionary."""
    for key in keys:
        value = row.get(key)
        if value is not None and value != "":
            return value
    return default


def normalize_severity(member):
    value = str(
        first_value(
            member,
            "severity",
            "risk_tier",
            "tier",
            "risk_level",
            default="Unknown",
        )
    ).strip().lower()

    if value in {"high", "critical", "very high"}:
        return "High"

    if value in {"medium", "moderate", "intermediate"}:
        return "Medium"

    if value in {"low", "minimal"}:
        return "Low"

    return "Unknown"


def audit_text(event):
    return " ".join(
        str(event.get(key, ""))
        for key in (
            "action",
            "event",
            "status",
            "message",
            "type",
        )
    ).lower()


def count_events(events, terms):
    return sum(
        any(term in audit_text(event) for term in terms)
        for event in events
    )


def format_timestamp(timestamp_ms):
    if not timestamp_ms:
        return "—"

    try:
        return datetime.fromtimestamp(
            timestamp_ms / 1000
        ).strftime("%Y-%m-%d %H:%M")
    except (TypeError, ValueError, OSError):
        return "—"


def heading(title, subtitle=None):
    st.subheader(title)

    if subtitle:
        st.markdown(
            f'<div class="section-note">{subtitle}</div>',
            unsafe_allow_html=True,
        )


def safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def make_pie(data, names, values, height=300, hole=0.48):
    """Display a reusable pie/donut chart."""
    if data.empty:
        st.info("No data is available for this chart.")
        return

    fig = px.pie(
        data,
        names=names,
        values=values,
        hole=hole,
    )

    fig.update_traces(
        textposition="inside",
        textinfo="percent+label",
    )

    fig.update_layout(
        height=height,
        margin=dict(t=10, b=10, l=10, r=10),
        legend_title_text="",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


def make_bar(data, x, y, height=300, color=None):
    """Display a reusable bar chart."""
    if data.empty:
        st.info("No data is available for this chart.")
        return

    kwargs = {}
    if color and color in data.columns:
        kwargs["color"] = color

    fig = px.bar(
        data,
        x=x,
        y=y,
        text=y,
        **kwargs,
    )

    fig.update_traces(textposition="outside")

    fig.update_layout(
        height=height,
        margin=dict(t=20, b=10, l=10, r=10),
        xaxis_title="",
        yaxis_title="",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("# 🩺 PredictMed")
    st.markdown("**CKD Risk Intelligence**")
    st.caption(
        "Patient risk, outreach, and model monitoring"
    )

    st.divider()

    page = st.radio(
        "Navigate",
        [
            "Overview",
            "Patient risk",
            "Outreach & audit",
            "Model monitoring",
        ],
    )

    st.divider()

    st.markdown("**Connected services**")
    st.caption(f"Backend: {API_URL}")
    st.caption(f"MLflow: {MLFLOW_URL}")

    if st.button(
        "↻ Refresh dashboard",
        use_container_width=True,
        type="primary",
    ):
        st.rerun()

    st.caption("Demonstration prototype")


# ============================================================
# LOAD BACKEND DATA
# ============================================================

members_data, members_error = get_api("/api/members")
audit_data, audit_error = get_api("/api/audit")
drift_data, drift_error = get_api("/api/drift")

members = (
    as_records(members_data)
    if members_data is not None
    else []
)

audit = (
    as_records(audit_data)
    if audit_data is not None
    else []
)


# ============================================================
# LOAD MLFLOW RUNS
# ============================================================

runs = []
mlflow_error = None

try:
    client = MlflowClient(
        tracking_uri=MLFLOW_URL
    )

    experiment = client.get_experiment_by_name(
        "predictmed"
    )

    if experiment:
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["attributes.start_time DESC"],
            max_results=50,
        )

except Exception as exc:
    mlflow_error = str(exc)


latest_run = runs[0] if runs else None

severity_values = [
    normalize_severity(member)
    for member in members
]

severity_counts = {
    "High": severity_values.count("High"),
    "Medium": severity_values.count("Medium"),
    "Low": severity_values.count("Low"),
    "Unknown": severity_values.count("Unknown"),
}

approved_count = count_events(
    audit, ["approv"]
)

queued_count = count_events(
    audit, ["queu"]
)

sent_count = count_events(
    audit, ["sent", "delivered"]
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">
            Kidney health · Risk analytics · Model operations
        </div>
        <h1>PredictMed Dashboard</h1>
        <p>
            Patient risk, outreach activity and model performance
            in one clear view.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

if members_error:
    st.warning(
        "Patient API unavailable. "
        "Check that the backend is running on port 8000. "
        f"Details: {members_error}"
    )


# ============================================================
# OVERVIEW PAGE
# ============================================================

if page == "Overview":

    heading(
        "At a glance",
        "A summary of current patient records and model performance.",
    )

    k1, k2, k3, k4 = st.columns(4)

    k1.metric("Patient records", len(members))
    k2.metric("High severity", severity_counts["High"])
    k3.metric("Medium severity", severity_counts["Medium"])
    k4.metric("Low severity", severity_counts["Low"])

    left, right = st.columns(2, gap="large")

    with left:
        heading(
            "Patient severity mix",
            "Distribution returned by the backend.",
        )

        severity_df = pd.DataFrame({
            "Severity": list(severity_counts.keys()),
            "Patients": list(severity_counts.values()),
        })

        severity_df = severity_df[
            severity_df["Patients"] > 0
        ]

        if not severity_df.empty:
            fig = px.pie(
                severity_df,
                names="Severity",
                values="Patients",
                hole=0.52,
                color="Severity",
                color_discrete_map={
                    "High": "#e45756",
                    "Medium": "#f2a541",
                    "Low": "#2a9d8f",
                    "Unknown": "#9aa8b2",
                },
            )

            fig.update_traces(
                textposition="inside",
                textinfo="percent+label",
            )

            fig.update_layout(
                height=310,
                showlegend=False,
                margin=dict(t=10, b=10, l=10, r=10),
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )
        else:
            st.info(
                "No patient records are available to chart yet."
            )

    with right:
        heading(
            "Outreach activity",
            "Counts of matching audit events.",
        )

        outreach_df = pd.DataFrame({
            "Event": [
                "Approval",
                "Queued",
                "Sent / delivered",
            ],
            "Count": [
                approved_count,
                queued_count,
                sent_count,
            ],
        })

        fig = px.bar(
            outreach_df,
            x="Event",
            y="Count",
            text="Count",
            color="Event",
            color_discrete_map={
                "Approval": "#4361ee",
                "Queued": "#f2a541",
                "Sent / delivered": "#2a9d8f",
            },
        )

        fig.update_traces(
            textposition="outside"
        )

        fig.update_layout(
            height=310,
            showlegend=False,
            margin=dict(t=15, b=10, l=10, r=10),
            xaxis_title="",
            yaxis_title="Events",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    heading(
        "Latest model run",
        "Metrics logged by the PredictMed training pipeline.",
    )

    if mlflow_error:
        st.error(
            f"MLflow connection error: {mlflow_error}"
        )

    elif latest_run:
        metrics = latest_run.data.metrics

        m1, m2, m3, m4 = st.columns(4)

        m1.metric(
            "Run status",
            latest_run.info.status.title(),
        )

        m2.metric(
            "ROC AUC",
            f"{metrics['auc']:.3f}"
            if "auc" in metrics
            else "N/A",
        )

        m3.metric(
            "Raw Brier",
            f"{metrics['brier_raw']:.4f}"
            if "brier_raw" in metrics
            else "N/A",
        )

        m4.metric(
            "Calibrated Brier",
            f"{metrics['brier_calibrated']:.4f}"
            if "brier_calibrated" in metrics
            else "N/A",
        )

        st.caption(
            f"Run: {latest_run.info.run_name or latest_run.info.run_id[:8]} "
            f"· Model: {latest_run.data.params.get('model_kind', 'Unknown')} "
            f"· Started: {format_timestamp(latest_run.info.start_time)}"
        )

    else:
        st.info(
            "No runs found in the predictmed experiment yet."
        )

    heading(
        "Service status",
        "Connection checks for your running services.",
    )

    s1, s2, s3 = st.columns(3)

    if members_error:
        s1.error("PredictMed API unavailable")
    else:
        s1.success("PredictMed API connected")

    if mlflow_error:
        s2.error("MLflow unavailable")
    else:
        s2.success("MLflow connected")

    if drift_error:
        s3.error("Drift endpoint unavailable")
    else:
        s3.success("Drift endpoint reachable")


# ============================================================
# PATIENT RISK PAGE
# ============================================================

elif page == "Patient risk":

    heading(
        "Patient risk registry",
        "Search, filter, review and export backend records.",
    )

    if members:

        rows = []

        for member in members:
            rows.append({
                "Patient / ID": first_value(
                    member,
                    "name",
                    "member_name",
                    "id",
                    "member_id",
                    default="Unknown",
                ),
                "Severity": normalize_severity(member),
                "Risk score": first_value(
                    member,
                    "risk_percent",
                    "risk_score",
                    "probability",
                    "risk",
                ),
                "eGFR": first_value(
                    member,
                    "egfr",
                    "eGFR",
                ),
                "DVI": first_value(
                    member,
                    "dvi",
                    "DVI",
                ),
                "Status": first_value(
                    member,
                    "status",
                    "approval_status",
                    default="Unknown",
                ),
            })

        df = pd.DataFrame(rows)

        f1, f2 = st.columns([1, 2])

        severity_filter = f1.selectbox(
            "Filter by severity",
            ["All", "High", "Medium", "Low", "Unknown"],
        )

        search_text = f2.text_input(
            "Search patient or ID",
            placeholder="Enter a name or ID...",
        )

        filtered = df.copy()

        if severity_filter != "All":
            filtered = filtered[
                filtered["Severity"] == severity_filter
            ]

        if search_text.strip():
            filtered = filtered[
                filtered["Patient / ID"]
                .astype(str)
                .str.contains(
                    search_text.strip(),
                    case=False,
                    na=False,
                )
            ]

        chart_left, chart_right = st.columns(
            [1, 1.3],
            gap="large",
        )

        with chart_left:
            heading("Filtered severity")

            counts = (
                filtered["Severity"]
                .value_counts()
                .rename_axis("Severity")
                .reset_index(name="Patients")
            )

            if not counts.empty:
                fig = px.pie(
                    counts,
                    names="Severity",
                    values="Patients",
                    hole=0.48,
                    color="Severity",
                    color_discrete_map={
                        "High": "#e45756",
                        "Medium": "#f2a541",
                        "Low": "#2a9d8f",
                        "Unknown": "#9aa8b2",
                    },
                )

                fig.update_layout(
                    height=270,
                    margin=dict(t=5, b=5, l=5, r=5),
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )
            else:
                st.info("No matching patients.")

        with chart_right:
            heading("Risk score by patient")

            plot_df = filtered.copy()

            plot_df["Risk score"] = pd.to_numeric(
                plot_df["Risk score"],
                errors="coerce",
            )

            plot_df = plot_df.dropna(
                subset=["Risk score"]
            )

            if not plot_df.empty:
                fig = px.bar(
                    plot_df,
                    x="Patient / ID",
                    y="Risk score",
                    color="Severity",
                    color_discrete_map={
                        "High": "#e45756",
                        "Medium": "#f2a541",
                        "Low": "#2a9d8f",
                        "Unknown": "#9aa8b2",
                    },
                )

                fig.update_layout(
                    height=270,
                    margin=dict(t=5, b=5, l=5, r=5),
                    xaxis_title="Patient",
                    yaxis_title="Risk score",
                    xaxis_tickangle=-25,
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )
            else:
                st.info(
                    "No numeric risk scores are available "
                    "for the current filter."
                )

        st.dataframe(
            filtered,
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            f"Showing {len(filtered)} of {len(df)} patient records."
        )

        st.download_button(
            "⬇️ Export patient records (CSV)",
            data=filtered.to_csv(
                index=False
            ).encode("utf-8"),
            file_name="predictmed_patient_risk.csv",
            mime="text/csv",
        )

    else:
        st.info(
            "No patient records returned by GET /api/members."
        )

        if members_error:
            st.error(
                f"API connection error: {members_error}"
            )


# ============================================================
# OUTREACH AND AUDIT PAGE
# ============================================================

elif page == "Outreach & audit":

    heading(
        "Outreach & audit",
        "Review workflow events and export the audit history.",
    )

    a1, a2, a3, a4 = st.columns(4)

    a1.metric("Audit events", len(audit))
    a2.metric("Approval events", approved_count)
    a3.metric("Queue events", queued_count)
    a4.metric("Sent / delivered", sent_count)

    status_df = pd.DataFrame({
        "Status": [
            "Approved",
            "Queued",
            "Sent / delivered",
        ],
        "Events": [
            approved_count,
            queued_count,
            sent_count,
        ],
    })

    chart1, chart2 = st.columns(
        [1, 1.2],
        gap="large",
    )

    with chart1:
        heading("Workflow share")

        nonzero = status_df[
            status_df["Events"] > 0
        ]

        if not nonzero.empty:
            fig = px.pie(
                nonzero,
                names="Status",
                values="Events",
                hole=0.5,
                color="Status",
                color_discrete_map={
                    "Approved": "#4361ee",
                    "Queued": "#f2a541",
                    "Sent / delivered": "#2a9d8f",
                },
            )

            fig.update_layout(
                height=280,
                margin=dict(t=5, b=5, l=5, r=5),
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )
        else:
            st.info(
                "No matching workflow events have been recorded."
            )

    with chart2:
        heading("Workflow counts")

        fig = px.bar(
            status_df,
            x="Status",
            y="Events",
            text="Events",
            color="Status",
            color_discrete_map={
                "Approved": "#4361ee",
                "Queued": "#f2a541",
                "Sent / delivered": "#2a9d8f",
            },
        )

        fig.update_traces(
            textposition="outside"
        )

        fig.update_layout(
            height=280,
            showlegend=False,
            margin=dict(t=5, b=5, l=5, r=5),
            xaxis_title="",
            yaxis_title="Events",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    heading(
        "Audit event history",
        "Search the full event payload returned by the API.",
    )

    if audit:
        audit_df = pd.DataFrame(audit)

        audit_search = st.text_input(
            "Search audit events",
            placeholder="Action, status, message or ID...",
        )

        filtered_audit = audit_df.copy()

        if audit_search.strip():
            mask = (
                filtered_audit
                .astype(str)
                .apply(
                    lambda col: col.str.contains(
                        audit_search.strip(),
                        case=False,
                        na=False,
                    )
                )
                .any(axis=1)
            )

            filtered_audit = filtered_audit[mask]

        st.dataframe(
            filtered_audit,
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            f"Showing {len(filtered_audit)} of "
            f"{len(audit_df)} audit events."
        )

        st.download_button(
            "⬇️ Download audit CSV",
            data=filtered_audit.to_csv(
                index=False
            ).encode("utf-8"),
            file_name="predictmed_audit.csv",
            mime="text/csv",
        )

    else:
        st.info(
            "No audit events returned by GET /api/audit."
        )

        if audit_error:
            st.error(
                f"Audit API connection error: {audit_error}"
            )

    st.markdown(
        """
        <div class="notice">
            <strong>Delivery status:</strong>
            Approval or queueing does not prove a message was sent.
            Only count a message as sent when a real delivery event
            is recorded by the system.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# MODEL MONITORING PAGE
# ============================================================

elif page == "Model monitoring":

    heading(
        "Model monitoring",
        "MLflow training history, evaluation metrics and drift status.",
    )

    c1, c2 = st.columns(2)

    if mlflow_error:
        c1.error("MLflow unavailable")
    else:
        c1.success("MLflow tracking connected")

    if drift_error:
        c2.error("Drift endpoint unavailable")
    else:
        c2.success("Drift endpoint reachable")

    if mlflow_error:
        st.error(
            f"MLflow connection error: {mlflow_error}"
        )

    elif runs:

        run_rows = []

        for run in runs:
            metrics = run.data.metrics

            run_rows.append({
                "Run": (
                    run.info.run_name
                    or run.info.run_id[:8]
                ),
                "Run ID": run.info.run_id[:12],
                "Status": run.info.status,
                "Started": format_timestamp(
                    run.info.start_time
                ),
                "Model": run.data.params.get(
                    "model_kind",
                    "Unknown",
                ),
                "ROC AUC": metrics.get("auc"),
                "Raw Brier": metrics.get("brier_raw"),
                "Calibrated Brier": metrics.get(
                    "brier_calibrated"
                ),
            })

        run_df = pd.DataFrame(run_rows)

        st.dataframe(
            run_df,
            use_container_width=True,
            hide_index=True,
        )

        latest_metrics = latest_run.data.metrics

        m1, m2, m3, m4 = st.columns(4)

        m1.metric(
            "Training runs",
            len(runs),
        )

        m2.metric(
            "Latest ROC AUC",
            f"{latest_metrics['auc']:.3f}"
            if "auc" in latest_metrics
            else "N/A",
        )

        m3.metric(
            "Raw Brier",
            f"{latest_metrics['brier_raw']:.4f}"
            if "brier_raw" in latest_metrics
            else "N/A",
        )

        m4.metric(
            "Calibrated Brier",
            f"{latest_metrics['brier_calibrated']:.4f}"
            if "brier_calibrated" in latest_metrics
            else "N/A",
        )

        chart_df = run_df[
            [
                "Run",
                "ROC AUC",
                "Raw Brier",
                "Calibrated Brier",
            ]
        ].copy()

        for column in (
            "ROC AUC",
            "Raw Brier",
            "Calibrated Brier",
        ):
            chart_df[column] = pd.to_numeric(
                chart_df[column],
                errors="coerce",
            )

        auc_df = chart_df[
            ["Run", "ROC AUC"]
        ].dropna()

        if not auc_df.empty:
            heading(
                "ROC AUC by run",
                "Higher is generally better. "
                "These results use the project's current evaluation data.",
            )

            fig = px.bar(
                auc_df,
                x="Run",
                y="ROC AUC",
                text="ROC AUC",
                color_discrete_sequence=["#087f8c"],
            )

            fig.update_traces(
                texttemplate="%{y:.3f}",
                textposition="outside",
            )

            fig.update_layout(
                height=300,
                margin=dict(t=15, b=10, l=10, r=10),
                yaxis_range=[0, 1],
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        brier_df = chart_df.melt(
            id_vars="Run",
            value_vars=[
                "Raw Brier",
                "Calibrated Brier",
            ],
            var_name="Metric",
            value_name="Score",
        ).dropna()

        if not brier_df.empty:
            heading(
                "Brier score comparison",
                "Lower scores are generally better.",
            )

            fig = px.bar(
                brier_df,
                x="Run",
                y="Score",
                color="Metric",
                barmode="group",
                color_discrete_map={
                    "Raw Brier": "#f2a541",
                    "Calibrated Brier": "#4361ee",
                },
            )

            fig.update_layout(
                height=300,
                margin=dict(t=10, b=10, l=10, r=10),
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

    else:
        st.info(
            "No runs found in the predictmed MLflow experiment. "
            "Run POST /api/retrain to create a training run."
        )

    heading(
        "Drift watch",
        "Current response from GET /api/drift.",
    )

    if drift_error:
        st.warning(
            f"Drift endpoint error: {drift_error}"
        )

    elif drift_data is not None:

        if isinstance(drift_data, dict):

            d1, d2, d3 = st.columns(3)

            drift_value = first_value(
                drift_data,
                "psi",
                "PSI",
                "population_stability_index",
            )

            drift_status = first_value(
                drift_data,
                "status",
                "state",
                "message",
                default="Available",
            )

            drift_version = first_value(
                drift_data,
                "model_version",
                "version",
                default="—",
            )

            parsed_psi = safe_float(drift_value)

            d1.metric(
                "PSI",
                f"{parsed_psi:.3f}"
                if parsed_psi is not None
                else "N/A",
            )

            d2.metric(
                "Status",
                str(drift_status),
            )

            d3.metric(
                "Model version",
                str(drift_version),
            )

        with st.expander("View full drift response"):
            st.json(drift_data)

    else:
        st.info(
            "No drift information returned by GET /api/drift."
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        <strong>Prototype notice:</strong>
        This dashboard is for demonstration and operational monitoring.
        The default model uses synthetic data. Outputs are not clinically
        validated and must not be used for medical decisions. Confirm risk
        definitions with qualified clinicians and protect patient information.
    </div>
    """,
    unsafe_allow_html=True,
)
