from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st


# ============================================================
# MODELLIFE | ADAPTIVE ML LIFECYCLE
# Professional black, cyan, blue and teal dashboard
# ============================================================

ROOT = Path(__file__).resolve().parent
REGISTRY_PATH = ROOT / "models" / "model_metadata.json"
PREDICTIONS_PATH = ROOT / "logs" / "predictions.csv"

st.set_page_config(
    page_title="ModelLife | Workspace",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# THEME AND STYLING
# ============================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

    
    :root {
    --bg: #1C1D21;
    --sidebar: #151619;
    --panel: #222429;
    --panel2: #292B31;
    --border: #383B43;
    --text: #F4F6FA;
    --muted: #A5A9B3;
    --cyan: #16C8F3;
    --blue: #398BFA;
    --teal: #20D6B2;
}

    
    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

   .stApp {
    background: #1C1D21;
    color: #F4F6FA;
}

    [data-testid="stHeader"] {
        background: rgba(5, 8, 13, 0.96);
    }

    [data-testid="stSidebar"] {
        background: var(--sidebar);
        border-right: 1px solid #1B2A39;
    }

    [data-testid="stSidebar"] > div:first-child {
        background: var(--sidebar);
    }

    .block-container {
        max-width: 1800px;
        padding-top: 1.1rem;
        padding-bottom: 2rem;
    }

    h1, h2, h3, h4 {
        font-family: 'Manrope', sans-serif !important;
        color: var(--text) !important;
        letter-spacing: -0.3px;
    }

    p, label, .stMarkdown {
        color: var(--text);
    }

    [data-testid="stCaptionContainer"] {
        color: var(--muted);
    }

    /* ---------------- SIDEBAR BRAND ---------------- */

    .ml-brand {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 9px 5px 0 4px;
    }

    .ml-brand-icon {
        font-size: 37px;
        line-height: 1;
        filter: drop-shadow(0 0 7px rgba(22, 200, 243, .2));
    }

    .ml-brand-name {
        font-family: 'Manrope', sans-serif;
        font-size: 27px;
        font-weight: 800;
        color: #F5FAFF;
        letter-spacing: -1.2px;
        line-height: 1.1;
    }

    .ml-brand-name span {
        color: var(--cyan);
    }

    .ml-brand-sub {
        color: #9FB2C5;
        font-size: 11px;
        margin-left: 47px;
        margin-top: -1px;
        margin-bottom: 20px;
    }

    .sidebar-divider {
        border-top: 1px solid #1B2A39;
        margin: 13px 0 14px;
    }

    /* Sidebar radio navigation is styled as a set of clickable cards */
    [data-testid="stSidebar"] [data-testid="stRadio"] > label {
        display: none;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] > div {
        gap: 7px;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        background: transparent;
        border: 1px solid transparent;
        border-radius: 11px;
        padding: 11px 10px;
        min-height: 51px;
        transition: background .15s ease, border .15s ease;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
        background: #101B28;
        border-color: #21384B;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] label p {
        font-size: 13px;
        font-weight: 600;
        color: #E6EDF5;
        margin: 0;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
        background: linear-gradient(100deg, #10283A, #102033);
        border: 1px solid #1D5268;
        box-shadow: inset 3px 0 0 var(--cyan);
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) p {
        color: #FFFFFF;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] input {
        accent-color: var(--cyan);
    }

    .sidebar-project {
        background: #0B1420;
        border: 1px solid #203244;
        border-radius: 12px;
        padding: 13px;
        margin-top: 14px;
        margin-bottom: 16px;
    }

    .sidebar-project-label {
        color: var(--cyan);
        font-size: 11px;
        font-weight: 600;
        letter-spacing: .3px;
    }

    .sidebar-project-name {
        color: #F5FAFF;
        font-size: 14px;
        font-weight: 700;
        margin-top: 4px;
    }

    .sidebar-project-path {
        color: #9AABBD;
        font-size: 10px;
        margin-top: 5px;
        overflow-wrap: anywhere;
    }

    .sidebar-status-heading {
        color: #A8BDCF;
        font-size: 12px;
        font-weight: 600;
        margin: 12px 0 6px;
    }

    .sidebar-status {
        color: #C0CEDB;
        font-size: 11px;
        line-height: 2;
    }

    /* ---------------- TOP BAR ---------------- */

    .topbar-note {
        color: #9FB1C3;
        font-size: 11px;
        text-align: right;
        padding-top: 9px;
    }

    .stTextInput input,
    .stTextArea textarea {
        background: #0B1420 !important;
        color: #F4F8FC !important;
        border: 1px solid #25394C !important;
        border-radius: 9px !important;
    }

    .stTextInput input::placeholder,
    .stTextArea textarea::placeholder {
        color: #8295A9 !important;
    }

    div[data-baseweb="select"] > div {
        background: #0B1420;
        border-color: #25394C;
        color: #F4F8FC;
    }

    /* ---------------- HERO ---------------- */

    .hero {
        background:
            radial-gradient(ellipse at 78% 50%, rgba(22, 200, 243, .14), transparent 36%),
            linear-gradient(110deg, #0D1B2A 0%, #09131F 62%, #0B1E29 100%);
        border: 1px solid #1D5268;
        border-radius: 15px;
        padding: 22px 24px;
        min-height: 174px;
        position: relative;
        overflow: hidden;
        margin-bottom: 12px;
    }

    .hero-kicker {
        color: #A6DDEB;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: .7px;
    }

    .hero-title {
        font-family: 'Manrope', sans-serif;
        font-size: 39px;
        line-height: 1.15;
        letter-spacing: -1.8px;
        font-weight: 800;
        color: #F7FAFE;
        margin-top: 5px;
    }

    .hero-title span {
        color: var(--cyan);
    }

    .hero-subtitle {
        color: #AFC6D9;
        font-size: 15px;
        margin-top: 5px;
    }

    .hero-feature-title {
        color: #DCEAF5;
        font-size: 12px;
        font-weight: 600;
        margin-bottom: 5px;
    }

    .hero-feature {
        color: #B8C9D9;
        font-size: 11px;
        line-height: 1.9;
    }

    .hero-feature b {
        color: var(--cyan);
        margin-right: 5px;
    }

    /* ---------------- KPI CARDS ---------------- */

    .kpi {
        background: linear-gradient(145deg, #101B28, #0A111A);
        border: 1px solid #23384B;
        border-radius: 12px;
        padding: 13px 13px 11px;
        min-height: 111px;
        margin-bottom: 11px;
    }

    .kpi-icon {
        float: left;
        display: inline-flex;
        justify-content: center;
        align-items: center;
        width: 40px;
        height: 40px;
        border-radius: 10px;
        background: #10283A;
        border: 1px solid #1C4259;
        color: var(--cyan);
        font-size: 20px;
        margin-right: 10px;
    }

    .kpi-label {
        color: #AABBCD;
        font-size: 11px;
        padding-top: 2px;
    }

    .kpi-value {
        color: #F5F9FD;
        font-family: 'Manrope', sans-serif;
        font-size: 23px;
        font-weight: 800;
        line-height: 1.25;
        margin-top: 3px;
    }

    .kpi-foot {
        color: #91A6BA;
        font-size: 10px;
        margin-top: 6px;
        clear: both;
    }

    /* ---------------- PANELS AND CHARTS ---------------- */

    .section-title {
        font-family: 'Manrope', sans-serif;
        color: #F0F5FA;
        font-size: 16px;
        font-weight: 700;
        margin: 7px 0 1px;
    }

    .section-subtitle {
        color: #91A6BA;
        font-size: 11px;
        margin-bottom: 8px;
    }

    .panel {
        background: #0A111A;
        border: 1px solid #203244;
        border-radius: 13px;
        padding: 13px;
        margin-bottom: 12px;
    }

    /* ---------------- ASSISTANT ---------------- */

    .assistant-panel {
        background: linear-gradient(150deg, #0D1825, #09111A);
        border: 1px solid #20394D;
        border-radius: 14px;
        padding: 15px;
        margin-bottom: 12px;
    }

    .assistant-heading {
        display: flex;
        align-items: center;
        gap: 10px;
        font-family: 'Manrope', sans-serif;
        font-size: 19px;
        font-weight: 800;
        color: #F3F8FC;
    }

    .assistant-icon {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 39px;
        height: 39px;
        border-radius: 11px;
        background: #10303C;
        border: 1px solid #1B5B68;
        color: var(--cyan);
        font-size: 19px;
    }

    .online {
        color: #A7B8C9;
        font-size: 10px;
        margin-left: auto;
        font-weight: 400;
    }

    .online b {
        color: #36D48D;
        margin-right: 4px;
    }

    .assistant-copy {
        color: #AABBCD;
        font-size: 12px;
        line-height: 1.7;
        margin: 12px 0;
    }

    .assistant-action {
        background: linear-gradient(110deg, #102236, #0E1A28);
        border: 1px solid #244158;
        border-radius: 10px;
        padding: 11px 12px;
        margin: 8px 0;
    }

    .assistant-action strong {
        color: #EAF3FA;
        font-size: 12px;
    }

    .assistant-action small {
        display: block;
        color: #9DB1C4;
        font-size: 10px;
        margin-top: 3px;
    }

    .assistant-hint {
        color: #8EA3B8;
        font-size: 10px;
        margin-top: 8px;
    }

    /* ---------------- PROJECT INSIGHTS ---------------- */

    .insight-card {
        background: #0D1925;
        border: 1px solid #20384A;
        border-radius: 11px;
        padding: 11px;
        min-height: 76px;
        margin-bottom: 8px;
    }

    .insight-number {
        color: var(--cyan);
        font-family: 'Manrope', sans-serif;
        font-size: 22px;
        font-weight: 800;
    }

    .insight-label {
        color: #A5B6C8;
        font-size: 10px;
        margin-top: 1px;
    }

    /* ---------------- TABLES / BUTTONS ---------------- */

    [data-testid="stDataFrame"] {
        border: 1px solid #203244;
        border-radius: 10px;
        overflow: hidden;
    }

    [data-testid="stMetric"] {
        background: #0D1825;
        border: 1px solid #203244;
        padding: 11px;
        border-radius: 10px;
    }

    [data-testid="stMetricLabel"] {
        color: #AABBCD !important;
    }

    [data-testid="stMetricValue"] {
        color: #F4F8FC !important;
        font-family: 'Manrope', sans-serif;
    }

    .stButton > button {
        background: linear-gradient(100deg, #087FA7, #0FAAC0);
        color: white;
        border: 1px solid #167D9B;
        border-radius: 9px;
        font-weight: 600;
        min-height: 36px;
    }

    .stButton > button:hover {
        background: #16C8F3;
        color: #041018;
        border-color: #16C8F3;
    }

    hr {
        border-color: #203244;
    }

    .footer {
        color: #71869B;
        font-size: 10px;
        text-align: center;
        padding: 9px 0 2px;
    }

    @media (max-width: 900px) {
        .hero-title { font-size: 29px; }
        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATA HELPERS
# ============================================================

def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON file safely."""
    if not path.exists():
        return {}

    try:
        with path.open("r", encoding="utf-8") as file:
            result = json.load(file)
        return result if isinstance(result, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV file safely."""
    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError):
        return pd.DataFrame()


def safe_float(value: Any) -> float | None:
    """Convert a value to float without crashing."""
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def get_registry_parts(data: dict[str, Any]):
    """
    Support common registry layouts:
    {
        "current_version": "v2",
        "models": {"v1": {...}, "v2": {...}}
    }
    or a list of model entries.
    """
    current = str(data.get("current_version", "Not set"))
    models = data.get("models", {})

    if isinstance(models, dict):
        entries = []
        for version, details in models.items():
            if isinstance(details, dict):
                entry = dict(details)
                entry.setdefault("version", version)
                entries.append(entry)
    elif isinstance(models, list):
        entries = [item for item in models if isinstance(item, dict)]
    else:
        entries = []

    active = {}
    for entry in entries:
        version = str(
            entry.get("version", entry.get("model_version", ""))
        )
        if version == current:
            active = entry
            break

    return current, active, entries


def get_metric(entry: dict[str, Any], metric_name: str):
    """Read a metric from nested or flat registry metadata."""
    metrics = entry.get("metrics", {})

    if isinstance(metrics, dict):
        return safe_float(metrics.get(metric_name))

    return safe_float(entry.get(metric_name))


def format_metric(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "—"
    return f"{value:.{digits}f}"


def count_python_files(paths) -> int:
    return sum(1 for path in paths if str(path).lower().endswith(".py"))


def make_plot_layout(fig, height: int = 300):
    """Apply the dashboard's dark chart style."""
    fig.update_layout(
        height=height,
        template="plotly_dark",
        paper_bgcolor="#0A111A",
        plot_bgcolor="#0A111A",
        font=dict(
            family="DM Sans, sans-serif",
            color="#DCE8F3",
            size=11,
        ),
        margin=dict(l=10, r=10, t=15, b=10),
        legend_title_text="",
    )
    fig.update_xaxes(
        showgrid=False,
        linecolor="#25384A",
        tickfont=dict(color="#AFC0D1"),
    )
    fig.update_yaxes(
        gridcolor="#1C2A38",
        zeroline=False,
        tickfont=dict(color="#AFC0D1"),
    )
    return fig


# Load actual local artifacts
registry = read_json(REGISTRY_PATH)
current_version, active_model, model_entries = get_registry_parts(registry)
predictions = read_csv(PREDICTIONS_PATH)


# ============================================================
# SIDEBAR — REFERENCE-STYLE LEFT NAVIGATION
# ============================================================

NAV_ITEMS = {
    "Overview": "▦",
    "My Projects": "▱",
    "AI Assistant": "▤",
    "Interview Coach": "♧",
    "ML Monitoring": "⌁",
    "Model Registry": "◈",
    "Tasks & Reports": "☷",
}

with st.sidebar:
    st.markdown(
        """
        <div class="ml-brand">
            <div class="ml-brand-icon">🧠</div>
            <div class="ml-brand-name">Model<span>Life</span></div>
        </div>
        <div class="ml-brand-sub">Adaptive ML Lifecycle</div>
        <div class="sidebar-divider"></div>
        """,
        unsafe_allow_html=True,
    )

    # The selected item is highlighted by the custom CSS above.
    page = st.radio(
        "Navigation",
        options=list(NAV_ITEMS.keys()),
        format_func=lambda item: f"{NAV_ITEMS[item]}    {item}",
        label_visibility="collapsed",
        key="main_navigation",
    )

    st.markdown(
        f"""
        <div class="sidebar-project">
            <div class="sidebar-project-label">CURRENT PROJECT</div>
            <div class="sidebar-project-name">📁 ModelLife</div>
            <div class="sidebar-project-path">{ROOT}</div>
        </div>
        <div class="sidebar-divider"></div>
        <div class="sidebar-status-heading">System Status</div>
        <div class="sidebar-status">
            {"🟢" if REGISTRY_PATH.exists() else "🟠"} &nbsp; Model registry
            {"available" if REGISTRY_PATH.exists() else "not found"}<br>
            {"🟢" if not predictions.empty else "🟠"} &nbsp; Prediction logs
            {"available" if not predictions.empty else "not found"}<br>
            {"🟢" if (ROOT / "src").exists() else "🟠"} &nbsp; Source folder
            {"available" if (ROOT / "src").exists() else "not found"}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# TOP SEARCH BAR
# ============================================================

top_search_col, top_status_col = st.columns([4, 1])

with top_search_col:
    search_text = st.text_input(
        "Search",
        placeholder="⌕  Search files, ask a question, or navigate...",
        label_visibility="collapsed",
        key="workspace_search",
    )

with top_status_col:
    st.markdown(
        '<div class="topbar-note">● Local workspace</div>',
        unsafe_allow_html=True,
    )

if search_text.strip():
    st.caption(
        "Search text entered. Navigation and file search are not connected "
        "to this field yet."
    )


# ============================================================
# SHARED PROJECT INSIGHTS
# ============================================================

def get_project_insights():
    """Read the project's existing analyzer output, if available."""
    try:
        from src.project_analyzer import get_project_inventory

        inventory = get_project_inventory(ROOT)
        return {
            "files": inventory.get("files_scanned", 0),
            "python": count_python_files(inventory.get("file_paths", [])),
            "technologies": len(inventory.get("technologies", [])),
            "modules": len(
                [
                    path for path in inventory.get("file_paths", [])
                    if str(path).replace("\\", "/").startswith("src/")
                ]
            ),
            "inventory": inventory,
        }
    except Exception:
        return {
            "files": 0,
            "python": 0,
            "technologies": 0,
            "modules": 0,
            "inventory": {},
        }


# ============================================================
# OVERVIEW PAGE
# ============================================================

if page == "Overview":

    # Hero banner
    hero_left, hero_right = st.columns([1.4, 1])

    with hero_left:
        st.markdown(
            """
            <div class="hero">
                <div class="hero-kicker">WELCOME TO</div>
                <div class="hero-title">Model<span>Life</span></div>
                <div class="hero-subtitle">
                    Detect. Learn. Validate. Evolve.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with hero_right:
        st.markdown(
            """
            <div class="hero">
                <div class="hero-feature-title">
                    YOUR COMPLETE ML PROJECT WORKSPACE
                </div>
                <div class="hero-feature">
                    <b>✓</b> Project analysis<br>
                    <b>✓</b> Interview preparation<br>
                    <b>✓</b> Model monitoring<br>
                    <b>✓</b> AI-powered assistance
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # KPI values are read from the local registry and prediction log.
    accuracy = get_metric(active_model, "accuracy")
    roc_auc = get_metric(active_model, "roc_auc")
    prediction_count = len(predictions)

    k1, k2, k3, k4 = st.columns(4)

    def render_kpi(column, icon, label, value, foot):
        with column:
            st.markdown(
                f"""
                <div class="kpi">
                    <div class="kpi-icon">{icon}</div>
                    <div class="kpi-label">{label}</div>
                    <div class="kpi-value">{value}</div>
                    <div class="kpi-foot">{foot}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    render_kpi(
        k1, "◈", "Active Model", current_version,
        str(active_model.get("model_type", active_model.get("algorithm", "Registered model")))
    )
    render_kpi(
        k2, "◎", "Accuracy", format_metric(accuracy),
        "Current registered model"
    )
    render_kpi(
        k3, "▥", "ROC-AUC", format_metric(roc_auc),
        "Current registered model"
    )
    render_kpi(
        k4, "▤", "Predictions Logged", f"{prediction_count:,}",
        "Records in local prediction log"
    )

    # Main content and right-side assistant panel
    center_col, right_col = st.columns([2.25, 1])

    with center_col:

        # Performance chart
        st.markdown(
            '<div class="section-title">Model Performance Trends</div>'
            '<div class="section-subtitle">'
            'Key evaluation metrics across registered model versions'
            '</div>',
            unsafe_allow_html=True,
        )

        chart_rows = []
        metric_names = [
            ("accuracy", "Accuracy"),
            ("precision", "Precision"),
            ("recall", "Recall"),
            ("f1", "F1 Score"),
            ("roc_auc", "ROC-AUC"),
        ]

        for entry in model_entries:
            version = str(
                entry.get("version", entry.get("model_version", "Unknown"))
            )
            for metric_key, metric_label in metric_names:
                metric_value = get_metric(entry, metric_key)
                if metric_value is not None:
                    chart_rows.append(
                        {
                            "Version": version,
                            "Metric": metric_label,
                            "Score": metric_value,
                        }
                    )

        if chart_rows:
            performance_df = pd.DataFrame(chart_rows)
            fig = px.line(
                performance_df,
                x="Version",
                y="Score",
                color="Metric",
                markers=True,
                range_y=[0, 1],
                color_discrete_map={
                    "Accuracy": "#318BFA",
                    "Precision": "#20D6B2",
                    "Recall": "#F5B942",
                    "F1 Score": "#F16C7A",
                    "ROC-AUC": "#16C8F3",
                },
            )
            fig.update_traces(line=dict(width=2.5), marker=dict(size=7))
            make_plot_layout(fig, 310)
            st.plotly_chart(
                fig,
                use_container_width=True,
                config={"displayModeBar": False},
            )
        else:
            st.info(
                "No versioned metrics are available in the model registry yet. "
                "Once metrics are saved, the performance chart will appear here."
            )

        # Prediction distribution
        st.markdown(
            '<div class="section-title">Prediction Distribution</div>'
            '<div class="section-subtitle">'
            'Logged prediction counts grouped by model version'
            '</div>',
            unsafe_allow_html=True,
        )

        if (
            not predictions.empty
            and "model_version" in predictions.columns
        ):
            version_counts = (
                predictions["model_version"]
                .fillna("Unknown")
                .astype(str)
                .value_counts()
                .rename_axis("Model Version")
                .reset_index(name="Predictions")
            )

            bar_fig = px.bar(
                version_counts,
                x="Model Version",
                y="Predictions",
                text="Predictions",
                color="Model Version",
                color_discrete_sequence=["#318BFA", "#20D6B2", "#16C8F3"],
            )
            bar_fig.update_traces(
                textposition="outside",
                marker_line_width=0,
            )
            make_plot_layout(bar_fig, 265)
            st.plotly_chart(
                bar_fig,
                use_container_width=True,
                config={"displayModeBar": False},
            )
        elif (
            not predictions.empty
            and "prediction" in predictions.columns
        ):
            counts = (
                predictions["prediction"]
                .astype(str)
                .value_counts()
                .rename_axis("Prediction")
                .reset_index(name="Count")
            )
            pie_fig = px.pie(
                counts,
                names="Prediction",
                values="Count",
                hole=0.58,
                color_discrete_sequence=["#16C8F3", "#20D6B2", "#F5B942"],
            )
            make_plot_layout(pie_fig, 265)
            st.plotly_chart(
                pie_fig,
                use_container_width=True,
                config={"displayModeBar": False},
            )
        else:
            st.info(
                "Prediction charts will appear after the prediction service "
                "writes records to the local log."
            )

        # Recent predictions
        st.markdown(
            '<div class="section-title">Recent Predictions</div>'
            '<div class="section-subtitle">'
            'Most recent records from the prediction log'
            '</div>',
            unsafe_allow_html=True,
        )

        if not predictions.empty:
            recent = predictions.tail(8).iloc[::-1].copy()
            st.dataframe(
                recent,
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info(
                "No prediction records found. Run the prediction service "
                "to populate this section."
            )

    with right_col:

        # AI Assistant panel
        st.markdown(
            """
            <div class="assistant-panel">
                <div class="assistant-heading">
                    <span class="assistant-icon">✧</span>
                    AI Assistant
                    <span class="online"><b>●</b> Local</span>
                </div>
                <div class="assistant-copy">
                    Explore your project, understand code, identify
                    improvements, debug errors, and prepare for interviews.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        assistant_actions = [
            (
                "⌕",
                "Analyze my code",
                "Understand project structure and implementation",
            ),
            (
                "⚒",
                "Find improvement areas",
                "Review possible enhancements and best practices",
            ),
            (
                "⚠",
                "Explain an error",
                "Work through a debugging question",
            ),
            (
                "♧",
                "Prepare for interviews",
                "Practice questions based on your project",
            ),
        ]

        for icon, title, description in assistant_actions:
            st.markdown(
                f"""
                <div class="assistant-action">
                    <strong>{icon} &nbsp; {title} &nbsp; ›</strong>
                    <small>{description}</small>
                </div>
                """,
                unsafe_allow_html=True,
            )

        assistant_question = st.text_input(
            "Ask about your project",
            placeholder="Ask about your project...",
            label_visibility="collapsed",
            key="overview_assistant_input",
        )

        if st.button(
            "➤  Prepare guidance",
            key="overview_assistant_button",
            use_container_width=True,
        ):
            if assistant_question.strip():
                st.session_state["pending_assistant_question"] = assistant_question
                st.session_state["show_assistant_guidance"] = True
            else:
                st.warning("Enter a question first.")

        if st.session_state.get("show_assistant_guidance"):
            st.info(
                "Question saved for this session. The live LLM chat is not "
                "connected yet; use the project analyzer and source files "
                "to verify answers."
            )
            st.write(
                st.session_state.get("pending_assistant_question", "")
            )

        st.markdown(
            '<div class="section-title">💡 Project Insights</div>',
            unsafe_allow_html=True,
        )

        insights = get_project_insights()

        ins1, ins2 = st.columns(2)

        with ins1:
            st.markdown(
                f"""
                <div class="insight-card">
                    <div class="insight-number">{insights["python"]}</div>
                    <div class="insight-label">Python Files</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with ins2:
            st.markdown(
                f"""
                <div class="insight-card">
                    <div class="insight-number">{insights["technologies"]}</div>
                    <div class="insight-label">Technologies</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        ins3, ins4 = st.columns(2)

        with ins3:
            st.markdown(
                f"""
                <div class="insight-card">
                    <div class="insight-number">{insights["files"]}</div>
                    <div class="insight-label">Files Scanned</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with ins4:
            st.markdown(
                f"""
                <div class="insight-card">
                    <div class="insight-number">{len(model_entries)}</div>
                    <div class="insight-label">Registry Entries</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if st.button(
            "View project details →",
            key="view_project_details",
            use_container_width=True,
        ):
            st.session_state["main_navigation"] = "My Projects"
            st.rerun()


# ============================================================
# MY PROJECTS PAGE
# ============================================================

elif page == "My Projects":

    st.title("📁 My Projects")
    st.caption(
        "Inspect the current repository, its Python files, and detected technologies."
    )

    if st.button("⌕  Analyze current project", key="analyze_project"):
        try:
            from src.project_analyzer import get_project_inventory

            with st.spinner("Scanning allowed project files..."):
                st.session_state["project_inventory"] = (
                    get_project_inventory(ROOT)
                )
        except Exception as error:
            st.error(f"Project analysis failed: {error}")

    inventory = st.session_state.get("project_inventory")

    if inventory:
        file_paths = inventory.get("file_paths", [])
        technologies = inventory.get("technologies", [])

        c1, c2, c3 = st.columns(3)
        c1.metric("Files scanned", inventory.get("files_scanned", 0))
        c2.metric("Python files", count_python_files(file_paths))
        c3.metric("Technologies", len(technologies))

        st.subheader("Detected technologies")
        if technologies:
            st.write("  •  " + "\n  •  ".join(technologies))
        else:
            st.write("No technologies detected.")

        st.subheader("Project file inventory")
        st.dataframe(
            pd.DataFrame({"File path": file_paths}),
            use_container_width=True,
            hide_index=True,
        )

        with st.expander("View technology-to-file mapping"):
            st.json(inventory.get("technology_files", {}))
    else:
        st.info(
            "Select **Analyze current project** to scan the repository. "
            "The analyzer is read-only and does not execute source files."
        )


# ============================================================
# AI ASSISTANT PAGE
# ============================================================

elif page == "AI Assistant":

    st.title("✧ AI Assistant")
    st.caption("Project-aware development workspace")

    st.markdown(
        """
        <div class="assistant-panel">
            <div class="assistant-heading">
                <span class="assistant-icon">✧</span>
                Your project assistant
            </div>
            <div class="assistant-copy">
                Use this workspace to prepare code-review questions,
                understand project structure, and organize debugging.
                A live language-model connection is a planned integration,
                not an active feature in this dashboard.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    assistant_task = st.selectbox(
        "What would you like help with?",
        [
            "Analyze my code",
            "Find improvement areas",
            "Explain an error",
            "Ask about my project",
        ],
        key="assistant_task",
    )

    assistant_input = st.text_area(
        "Describe your question",
        placeholder=(
            "Example: Explain how my drift detection and candidate "
            "retraining workflow operates."
        ),
        height=130,
        key="assistant_input",
    )

    if st.button("Prepare guidance", key="assistant_submit"):
        if not assistant_input.strip():
            st.warning("Enter a question to continue.")
        else:
            st.session_state["assistant_request"] = {
                "task": assistant_task,
                "question": assistant_input,
            }
            st.session_state["assistant_request_ready"] = True

    if st.session_state.get("assistant_request_ready"):
        request = st.session_state["assistant_request"]

        st.subheader("Your request")
        st.write(f"**Task:** {request['task']}")
        st.write(f"**Question:** {request['question']}")

        st.subheader("Suggested next steps")
        st.markdown(
            """
            1. Open **My Projects** and run the project analyzer.
            2. Locate the relevant source file and related tests.
            3. Read the implementation and check its actual behavior.
            4. Run the appropriate tests before accepting a code change.
            """
        )
        st.info(
            "This page currently organizes your request and provides "
            "a workflow. It does not generate an LLM answer yet."
        )


# ============================================================
# INTERVIEW COACH PAGE
# ============================================================

elif page == "Interview Coach":

    st.title("🎓 Interview Coach")
    st.caption(
        "Generate project-aware interview questions and code-logic questions "
        "from your current repository."
    )

    st.markdown(
        """
        <div class="assistant-panel">
            <div class="assistant-heading">
                <span class="assistant-icon">✧</span>
                Practice with your own project
            </div>
            <div class="assistant-copy">
                Project answers cite documentation and source-level evidence.
                Code-logic questions are extracted from Python syntax structure
                without executing your source files. Source analysis does not
                establish behavior that has not been exercised.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    generation_type = st.radio(
        "Question source",
        options=[
            "Project interview",
            "Code logic",
            "Both",
        ],
        horizontal=True,
        key="interview_generation_type",
    )

    difficulty_filter = st.selectbox(
        "Difficulty",
        options=["All", "Beginner", "Intermediate", "Advanced"],
        key="interview_difficulty_filter",
    )

    if st.button(
        "⚡ Generate interview questions",
        key="generate_interview_questions",
        use_container_width=True,
    ):
        generated_project_questions = []
        generated_code_questions = []
        generation_errors = []

        if generation_type in ("Project interview", "Both"):
            try:
                from src.interview_generator import generate_interview_questions

                result = generate_interview_questions(str(ROOT))
                grouped = result.get("questions", {})
                for category, items in grouped.items():
                    for item in items:
                        normalized = dict(item)
                        normalized.setdefault("category", category.replace("_", " ").title())
                        normalized.setdefault("source_type", "project")
                        generated_project_questions.append(normalized)

                st.session_state["generated_project_questions"] = generated_project_questions
                st.session_state["generated_project_name"] = result.get(
                    "project_name", ROOT.name
                )
            except Exception as error:
                generation_errors.append(
                    f"Project interview generator: {type(error).__name__}: {error}"
                )

        if generation_type in ("Code logic", "Both"):
            try:
                from src.code_question_generator import generate_function_questions

                generated_code_questions = generate_function_questions(str(ROOT))
                st.session_state["generated_code_questions"] = generated_code_questions
            except Exception as error:
                generation_errors.append(
                    f"Code logic generator: {type(error).__name__}: {error}"
                )

        st.session_state["interview_generation_errors"] = generation_errors
        st.session_state["interview_last_generation_type"] = generation_type

        if generation_errors:
            for message in generation_errors:
                st.error(message)

        total_generated = (
            len(st.session_state.get("generated_project_questions", []))
            if generation_type == "Project interview"
            else len(st.session_state.get("generated_code_questions", []))
            if generation_type == "Code logic"
            else len(st.session_state.get("generated_project_questions", []))
            + len(st.session_state.get("generated_code_questions", []))
        )

        if total_generated:
            st.success(f"Generated {total_generated} question(s).")

    project_questions = st.session_state.get("generated_project_questions", [])
    code_questions = st.session_state.get("generated_code_questions", [])
    last_generation_type = st.session_state.get("interview_last_generation_type")

    # Only show the question collections requested by the most recent generation.
    if last_generation_type == "Project interview":
        all_questions = [
            {**item, "_display_type": "Project interview"}
            for item in project_questions
        ]
    elif last_generation_type == "Code logic":
        all_questions = [
            {**item, "_display_type": "Code logic"}
            for item in code_questions
        ]
    elif last_generation_type == "Both":
        all_questions = [
            {**item, "_display_type": "Project interview"}
            for item in project_questions
        ] + [
            {**item, "_display_type": "Code logic"}
            for item in code_questions
        ]
    else:
        all_questions = []

    if st.session_state.get("interview_generation_errors") and not all_questions:
        st.warning(
            "No questions are available yet. Check that both generator modules "
            "exist under src/ and run them from the project root to inspect errors."
        )

    if all_questions:
        def normalized_difficulty(item):
            value = str(item.get("difficulty", "Intermediate")).strip().lower()
            if value in {"beginner", "easy"}:
                return "Beginner"
            if value in {"advanced", "hard", "expert"}:
                return "Advanced"
            return "Intermediate"

        available_categories = sorted(
            {
                str(item.get("category", "General")).replace("_", " ").title()
                for item in all_questions
            }
        )

        filter_col1, filter_col2 = st.columns(2)
        with filter_col1:
            chosen_category = st.selectbox(
                "Filter by category",
                options=["All categories"] + available_categories,
                key="interview_category_filter",
            )
        with filter_col2:
            st.caption(f"{len(all_questions)} generated question(s) before filtering")

        filtered_questions = []
        for item in all_questions:
            category = str(item.get("category", "General")).replace("_", " ").title()
            difficulty = normalized_difficulty(item)

            if chosen_category != "All categories" and category != chosen_category:
                continue
            if difficulty_filter != "All" and difficulty != difficulty_filter:
                continue
            filtered_questions.append(item)

        st.markdown(
            f'<div class="section-title">Practice questions ({len(filtered_questions)})</div>'
        '<div class="section-subtitle">Open a question to review its answer versions and source evidence.</div>',
            unsafe_allow_html=True,
        )

        if not filtered_questions:
            st.info("No questions match these filters. Try a different category or difficulty.")
        else:
            def keep_selected_question_open(question_key):
                st.session_state["interview_expanded_question"] = question_key

            for index, item in enumerate(filtered_questions, start=1):
                question = item.get("question", "Practice question")
                category = str(item.get("category", "General")).replace("_", " ").title()
                difficulty = normalized_difficulty(item)
                display_type = item.get("_display_type", "Interview")
                expander_title = f"{index}. {question}"
                expander_key = f"{display_type}:{question}"
                expanded = (
                    st.session_state.get("interview_expanded_question")
                    == expander_key
                )

                with st.expander(expander_title, expanded=expanded):
                    meta_col1, meta_col2 = st.columns(2)
                    meta_col1.caption(f"Type: {display_type}")
                    meta_col2.caption(f"Category: {category} · {difficulty}")

                    why_asked = item.get("why_asked")
                    if why_asked:
                        st.markdown("**Why an interviewer may ask this**")
                        st.write(why_asked)

                    
                    answer_guidance = (
                        item.get("answer_guidance")
                        or item.get("annotation_explanation")
                    )

                    answer_versions = {
                        "Short": item.get("short_answer"),
                        "Medium": item.get("medium_answer"),
                        "Long / Detailed": item.get("detailed_answer"),
                    }

                    available_versions = {
                        label: answer
                        for label, answer in answer_versions.items()
                        if answer
                    }

                    if available_versions:
                        st.markdown("### Evidence-based interview answer")

                        selected_length = st.radio(
                            "Choose answer length",
                            options=list(available_versions.keys()),
                            horizontal=True,
                            key=f"answer_length_{expander_key}",
                            on_change=keep_selected_question_open,
                            args=(expander_key,),
                        )

                        st.write(available_versions[selected_length])

                    else:
                        st.info(
                            "No interview-ready answer is available for this "
                            "question. Review the source evidence for the "
                            "current implementation limits."
                        )



                    evidence = item.get("evidence", [])
                    if evidence:
                        with st.expander("Source evidence", expanded=False):
                            if isinstance(evidence, (list, tuple)):
                                for evidence_item in evidence:
                                    st.code(str(evidence_item), language="text")
                            else:
                                st.code(str(evidence), language="text")

                    debug_details = {
                        "Extracted guidance": answer_guidance,
                        "Parameters": item.get("parameters"),
                        "Return expressions": item.get("return_expressions"),
                        "Control-flow summary": item.get("control_flow_summary"),
                    }
                    if any(value for value in debug_details.values()):
                        with st.expander("Developer details", expanded=False):
                            for label, value in debug_details.items():
                                if value:
                                    st.markdown(f"**{label}**")
                                    st.write(value)

                    follow_ups = item.get("follow_ups", [])
                    if follow_ups:
                        st.markdown("**Possible follow-up questions**")
                        for follow_up in follow_ups:
                            st.write(f"• {follow_up}")

                    status_note = item.get("status")
                    if status_note:
                        st.caption(status_note)

                    answer_key = f"interview_answer_{display_type}_{index}"
                    saved_key = f"interview_saved_{display_type}_{index}"
                    practice_answer = st.text_area(
                        "Practice your answer",
                        placeholder="Write your answer here...",
                        key=answer_key,
                        height=110,
                    )

                    if st.button(
                        "Save practice answer",
                        key=f"save_{display_type}_{index}",
                    ):
                        st.session_state[saved_key] = practice_answer
                        st.success("Answer saved for this session.")

                    if saved_key in st.session_state:
                        st.caption("A practice answer is saved for this question in this session.")

        export_questions = []
        for item in filtered_questions:
            export_questions.append(
                {
                    key: value
                    for key, value in item.items()
                    if key != "_display_type"
                }
            )

        st.download_button(
            "⬇️ Export filtered questions as JSON",
            data=json.dumps(export_questions, ensure_ascii=False, indent=2, default=str),
            file_name="modellife_interview_questions.json",
            mime="application/json",
            use_container_width=True,
        )

    else:
        st.info(
            "Choose a question source and click **Generate interview questions**. "
            "The generators run locally against the current repository."
        )

    st.warning(
        "Answers are generated from scanned documentation and source evidence. "
        "Static analysis cannot confirm runtime behavior or personal experience. "
        "Live AI answer evaluation and spoken mock interviews are not connected yet."
    )


# ============================================================
# ML MONITORING PAGE
# ============================================================

elif page == "ML Monitoring":

    st.title("📈 ML Monitoring")
    st.caption(
        "Review the locally recorded predictions and model lifecycle information."
    )

    total_predictions = len(predictions)

    completed_count = 0
    pending_count = 0

    if not predictions.empty and "actual" in predictions.columns:
        actual_values = predictions["actual"]
        completed_count = int(actual_values.notna().sum())
        pending_count = total_predictions - completed_count

    m1, m2, m3 = st.columns(3)
    m1.metric("Predictions logged", total_predictions)
    m2.metric("Actual outcomes recorded", completed_count)
    m3.metric("Pending outcomes", pending_count)

    st.subheader("Prediction records")

    if not predictions.empty:
        st.dataframe(
            predictions.tail(100).iloc[::-1],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info(
            "No prediction log was found. Run the prediction service "
            "to generate local monitoring records."
        )

    st.subheader("Monitoring workflow")
    st.markdown(
        """
        - **Prediction logging:** stores prediction details and model version.
        - **Outcome tracking:** actual labels can be added when available.
        - **Performance monitoring:** evaluates completed predictions.
        - **Drift detection:** checks incoming data against reference data.
        - **Retraining review:** evaluates a candidate before promotion.
        """
    )

    st.caption(
        "This page displays local records; it is not a live streaming or "
        "production monitoring service."
    )


# ============================================================
# MODEL REGISTRY PAGE
# ============================================================

elif page == "Model Registry":

    st.title("◈ Model Registry")
    st.caption(
        "Review registered model versions and their recorded evaluation metrics."
    )

    st.metric("Current registered version", current_version)

    if model_entries:
        rows = []

        for entry in model_entries:
            rows.append(
                {
                    "Version": entry.get(
                        "version",
                        entry.get("model_version", "Unknown"),
                    ),
                    "Status": entry.get("status", "—"),
                    "Accuracy": get_metric(entry, "accuracy"),
                    "Precision": get_metric(entry, "precision"),
                    "Recall": get_metric(entry, "recall"),
                    "F1": get_metric(entry, "f1"),
                    "ROC-AUC": get_metric(entry, "roc_auc"),
                }
            )

        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True,
        )

        version_options = [
            str(
                entry.get("version", entry.get("model_version", "Unknown"))
            )
            for entry in model_entries
        ]

        selected_version = st.selectbox(
            "Inspect a registered version",
            version_options,
            key="registry_version",
        )

        selected_entry = next(
            (
                entry
                for entry in model_entries
                if str(
                    entry.get("version", entry.get("model_version", "Unknown"))
                ) == selected_version
            ),
            {},
        )

        st.subheader(f"Version {selected_version} details")

        r1, r2, r3, r4, r5 = st.columns(5)
        r1.metric("Accuracy", format_metric(get_metric(selected_entry, "accuracy")))
        r2.metric("Precision", format_metric(get_metric(selected_entry, "precision")))
        r3.metric("Recall", format_metric(get_metric(selected_entry, "recall")))
        r4.metric("F1", format_metric(get_metric(selected_entry, "f1")))
        r5.metric("ROC-AUC", format_metric(get_metric(selected_entry, "roc_auc")))

        with st.expander("View registry metadata"):
            st.json(registry)

    else:
        st.info(
            "No model entries were found. Check models/model_metadata.json "
            "and confirm that the registry contains registered versions."
        )


# ============================================================
# TASKS AND REPORTS PAGE
# ============================================================

elif page == "Tasks & Reports":

    st.title("☷ Tasks & Reports")
    st.caption(
        "Check whether key project files and folders are present."
    )

    checks = [
        ("Project analyzer", ROOT / "src" / "project_analyzer.py"),
        ("Interview question bank", ROOT / "src" / "interview_question_bank.py"),
        ("Code question generator", ROOT / "src" / "code_question_generator.py"),
        ("Model registry", REGISTRY_PATH),
        ("Prediction log", PREDICTIONS_PATH),
    ]

    for label, path in checks:
        exists = path.exists()
        st.write(
            f"{'🟢' if exists else '⚪'} **{label}:** "
            f"{'Available' if exists else 'Not found'}"
        )

    st.divider()
    st.subheader("Project folders")

    for folder in ["src", "tests", "data", "models", "logs"]:
        folder_path = ROOT / folder
        exists = folder_path.exists()
        st.write(
            f"{'🟢' if exists else '⚪'} `{folder}/` — "
            f"{'Present' if exists else 'Not created'}"
        )

    st.divider()
    st.subheader("Lifecycle overview")

    lifecycle_steps = [
        ("Data ingestion and preprocessing", "Prepare and validate incoming data"),
        ("Model training and evaluation", "Train and measure model performance"),
        ("Drift detection", "Compare incoming data with reference data"),
        ("Candidate retraining", "Train and evaluate a candidate when appropriate"),
        ("Human approval and registry", "Approve or reject candidate promotion"),
        ("Prediction and performance logs", "Track predictions and recorded outcomes"),
    ]

    for step, description in lifecycle_steps:
        st.markdown(f"**• {step}**")
        st.caption(description)

    st.caption(
        "This page checks file presence and displays workflow notes. "
        "It is not a complete automated test report."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div class="footer">
        ModelLife · Adaptive ML Lifecycle · Local project workspace
    </div>
    """,
    unsafe_allow_html=True,
)
