
import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import tempfile
from pathlib import Path
from academic_experiments import run_cross_validation, run_federated_cross_validation
from feature_schema import FEATURES
from app_services import (
    Principal,
    all_hospital_statuses,
    audit_log,
    authenticate,
    cancel_federated_round,
    current_model,
    dp_privacy_accounting,
    hospital_status,
    initialize as initialize_app,
    list_rounds,
    make_hospital_predictions,
    prediction_history,
    privacy_description,
    review_candidate,
    start_federated_round,
    submit_hospital_update,
)
from dp_accountant import DPAccountant
from dashboard_charts import (
    fedavg_training_history_chart,
    format_comparison_for_display,
    horizontal_grouped_count_chart,
    horizontal_metric_chart,
)
from inference import MODEL_ARTIFACTS, predict_probabilities, validate_feature_frame
from local_demo import is_local_demo_request, local_demo_mode_enabled
from workspace_access import page_is_allowed, pages_for_role
from live_evaluation import (
    run_iid_noniid_validation,
    run_live_model_comparison,
)
from retraining import (
    ACTIVE_MODEL_PATH,
    CANDIDATE_DIR,
    approve_baseline_model,
    approve_candidate,
    create_candidate,
    pending_confirmed_records,
    register_confirmed_records,
)

# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="Federated Breast Cancer Research",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE = Path(__file__).resolve().parent
METRICS = BASE / "metrics"
MODELS = BASE / "saved_models"

st.markdown("""
<style>
:root {
    color-scheme: light;
    --research-navy: #102a43;
    --research-ink: #243b53;
    --research-muted: #40566b;
    --research-teal: #087e8b;
    --research-teal-dark: #06636d;
    --research-border: #d9e2ec;
    --research-surface: #ffffff;
    --research-canvas: #f4f7fb;
}

.stApp {
    background: var(--research-canvas);
    color: var(--research-ink);
    font-size: 1rem;
}

[data-testid="stMainBlockContainer"] {
    max-width: 1440px;
    padding-top: 1.35rem;
    padding-bottom: 2rem;
}

[data-testid="stSidebar"] {
    background: var(--research-navy);
    border-right: 1px solid #243f59;
    color: #f4f8fb;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
[data-testid="stSidebar"] [data-testid="stRadio"] label {
    color: #f4f8fb;
}

[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
    font-size: 0.9rem;
}

[data-testid="stSidebar"] [data-testid="stAlert"] {
    background: #193b54;
    border: 1px solid #426178;
}

[data-testid="stSidebar"] button:not([data-testid^="stBaseButton-primary"]) {
    background: #193b54;
    border-color: #7892a7;
    color: #f4f8fb;
}

[data-testid="stSidebar"] button:not([data-testid^="stBaseButton-primary"]):hover:not(:disabled) {
    background: #24516b;
    border-color: #b7cad7;
    color: #ffffff;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label {
    min-height: 2.9rem;
    padding: 0.45rem 0.55rem;
    border-radius: 8px;
    font-weight: 600;
    font-size: 0.95rem;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
    background: #087e8b;
}

[data-testid="stButton"] button,
[data-testid="stFormSubmitButton"] button,
[data-testid="stDownloadButton"] button {
    min-width: 9rem;
    min-height: 2.9rem;
    border-radius: 8px;
    font-weight: 650;
    font-size: 1rem;
    line-height: 1.35;
}

[data-testid="stFileUploader"] button {
    min-width: 9rem;
    min-height: 2.9rem;
    border-radius: 8px;
    font-size: 1rem;
    font-weight: 650;
}

button[data-testid^="stBaseButton"] {
    min-width: 9rem;
    min-height: 2.9rem;
    font-size: 1rem;
    line-height: 1.35;
}

[data-testid="stDataFrame"] button {
    min-width: 2.75rem;
    min-height: 2.75rem;
    border-radius: 7px;
    font-size: 1rem;
}

[data-testid^="stBaseButton-primary"]:not(:disabled):not([aria-disabled="true"]) {
    background: #07545c !important;
    border: 1px solid #07545c !important;
    color: #ffffff !important;
    font-weight: 700 !important;
}

[data-testid^="stBaseButton-primary"]:not(:disabled):not([aria-disabled="true"]) *,
[data-testid^="stBaseButton-primary"]:not(:disabled):not([aria-disabled="true"]) p {
    color: #ffffff !important;
    font-weight: 700 !important;
}

[data-testid^="stBaseButton-primary"]:not(:disabled):not([aria-disabled="true"]):hover {
    background: #04464d !important;
    border-color: #04464d !important;
}

[data-testid^="stBaseButton-secondary"]:not(:disabled):not([aria-disabled="true"]) {
    background: #ffffff !important;
    border: 1px solid #70879a !important;
    color: #102a43 !important;
    font-weight: 650 !important;
}

[data-testid^="stBaseButton-secondary"]:not(:disabled):not([aria-disabled="true"]) *,
[data-testid^="stBaseButton-secondary"]:not(:disabled):not([aria-disabled="true"]) p {
    color: #102a43 !important;
}

[data-testid^="stBaseButton-secondary"]:not(:disabled):not([aria-disabled="true"]):hover {
    background: #edf3f7 !important;
    border-color: #40566b !important;
}

[data-testid^="stBaseButton"]:disabled,
[data-testid^="stBaseButton"][aria-disabled="true"] {
    background: #e5eaf0 !important;
    border: 1px solid #aab8c5 !important;
    color: #536575 !important;
    opacity: 1;
    cursor: not-allowed;
}

[data-testid^="stBaseButton"]:disabled *,
[data-testid^="stBaseButton"][aria-disabled="true"] * {
    background: transparent !important;
    color: #536575 !important;
    font-weight: 600 !important;
}

[data-testid="stSelectbox"] [role="option"],
[data-testid="stMultiSelect"] [role="option"] {
    background: #ffffff;
    color: #102a43;
    font-size: 1rem;
}

[data-testid="stCheckbox"] input,
[data-testid="stRadio"] input {
    accent-color: #07545c;
}

:where(a, button, input, textarea, summary, [role="combobox"], [role="radio"], [role="checkbox"], [role="button"]):focus-visible,
[data-testid="stCheckbox"] label:has(input:focus-visible),
[data-testid="stRadio"] label:has(input:focus-visible),
[data-testid="stFileUploader"] section:focus-within {
    outline: 3px solid #a34700 !important;
    outline-offset: 2px;
    box-shadow: none !important;
}

h1, h2, h3, h4 {
    color: var(--research-navy);
    letter-spacing: -0.025em;
}

h1 {
    font-size: 1.85rem;
    font-weight: 700;
}

h2 {
    font-size: 1.45rem;
}

h3 {
    font-size: 1.15rem;
}

[data-testid="stVerticalBlock"] > [data-testid="stVerticalBlockBorderWrapper"] {
    border-color: var(--research-border);
    border-radius: 12px;
    background: var(--research-surface);
    box-shadow: 0 2px 8px rgb(16 42 67 / 4%);
}

.st-key-login-card {
    max-width: 560px;
    margin: 1.5rem auto 0;
    padding: 1.5rem;
    background: var(--research-surface);
    border: 1px solid var(--research-border);
    border-radius: 14px;
    box-shadow: 0 8px 24px rgb(16 42 67 / 8%);
}

.st-key-login-card [data-testid="stTextInput"] input {
    min-height: 2.8rem;
}

[data-testid="stCaptionContainer"],
[data-testid="stMarkdownContainer"] p {
    color: var(--research-muted);
}

[data-testid="stCaptionContainer"] {
    font-size: 0.92rem;
    line-height: 1.5;
}

[data-testid="stWidgetLabel"] p,
[data-testid="stCheckbox"] label,
[data-testid="stRadio"] label {
    color: #243b53;
    font-size: 1rem;
    line-height: 1.4;
}

[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
[data-testid="stSidebar"] [data-testid="stRadio"] label {
    color: #f4f8fb;
}

[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input,
[data-testid="stTextArea"] textarea,
[data-testid="stDateInput"] input,
[data-testid="stSelectbox"] [role="combobox"],
[data-testid="stMultiSelect"] [role="combobox"] {
    min-height: 2.75rem;
    background: #ffffff !important;
    color: #102a43 !important;
    border-color: #70879a !important;
    border-radius: 8px;
    font-size: 1rem;
}

[data-testid="stFileUploader"] section {
    background: #ffffff;
    color: #243b53;
    border: 1px dashed #70879a;
    border-radius: 10px;
}

[data-testid="stFileUploader"] small,
[data-testid="stFileUploader"] [data-testid="stMarkdownContainer"] p {
    color: #40566b !important;
    font-size: 0.92rem;
}

[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary p {
    color: #102a43 !important;
    font-size: 1rem;
    font-weight: 600;
}

[data-testid="stMetric"] {
    min-height: 94px;
    padding: 0.8rem 0.9rem;
    background: var(--research-surface);
    border: 1px solid var(--research-border);
    border-top: 2px solid var(--research-teal);
    border-radius: 12px;
    box-shadow: 0 1px 5px rgb(16 42 67 / 5%);
}

[data-testid="stMetricLabel"] {
    color: var(--research-muted);
    font-weight: 600;
}

[data-testid="stMetricValue"] {
    color: var(--research-navy);
    font-weight: 750;
}

[data-testid="stDataFrame"],
[data-testid="stTable"] {
    border: 1px solid var(--research-border);
    border-radius: 10px;
    overflow: hidden;
}

[data-testid="stAlert"] {
    border-radius: 10px;
}

[data-testid="stExpander"] {
    background: var(--research-surface);
    border: 1px solid var(--research-border);
    border-radius: 10px;
}

[data-testid="stSubheader"] {
    margin-top: 1.6rem;
}

.research-hero {
    padding: 1.1rem 1.35rem;
    margin: 0.35rem 0 0.9rem;
    background: var(--research-navy);
    border: 1px solid #26445c;
    border-radius: 12px;
}

.research-hero h2,
.research-hero p {
    color: #ffffff !important;
    margin: 0;
}

.research-hero h2 {
    font-size: 1.55rem;
    font-weight: 700;
}

.research-hero p {
    margin-top: 0.4rem;
    color: #e1edf3 !important;
}

.home-hero {
    padding: 1.65rem 1.8rem;
    margin: 0.35rem 0 1.2rem;
    background: var(--research-navy);
    border: 1px solid #26445c;
    border-radius: 14px;
}

.home-hero h1,
.home-hero p {
    color: #ffffff !important;
    margin: 0;
}

.home-hero h1 {
    font-size: 2rem;
    font-weight: 750;
    line-height: 1.2;
    overflow-wrap: anywhere;
}

.home-hero p {
    max-width: 900px;
    margin-top: 0.65rem;
    color: #e1edf3 !important;
    font-size: 1rem;
    line-height: 1.6;
}

.navigation-card {
    min-height: 116px;
}

.workflow-step {
    min-height: 104px;
    padding: 0.85rem;
    background: var(--research-surface);
    border: 1px solid var(--research-border);
    border-radius: 10px;
}

.workflow-step strong {
    display: block;
    margin-bottom: 0.45rem;
    color: var(--research-teal-dark);
    font-size: 0.94rem;
}

.workflow-step span {
    color: var(--research-ink);
    line-height: 1.5;
    font-size: 0.92rem;
}

.status-chip {
    display: inline-block;
    padding: 0.22rem 0.55rem;
    color: var(--research-teal-dark);
    background: #e8f4f4;
    border: 1px solid #b8dcdd;
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 650;
}

.research-note {
    padding: 0.8rem 1rem;
    background: #edf5f8;
    border-left: 3px solid var(--research-teal);
    border-radius: 7px;
    color: var(--research-ink);
}

@media (max-width: 900px) {
    [data-testid="stMainBlockContainer"] {
        padding: 1.1rem 1rem 2rem;
    }

    .workflow-step {
        min-height: auto;
    }

    .st-key-login-card {
        max-width: none;
        margin: 0;
        padding: 1.1rem;
    }
}
</style>
""", unsafe_allow_html=True)

local_demo_enabled = local_demo_mode_enabled(os.environ)
if local_demo_enabled and not is_local_demo_request(
    st.get_option("server.address"),
    st.context.headers,
):
    st.error(
        "Local demo mode was requested, but this request is not using a direct "
        "loopback-only connection. Authentication remains required."
    )
    st.stop()

initialize_app()

if local_demo_enabled:
    admin_password = os.environ.get("FL_ADMIN_PASSWORD")
    if not admin_password:
        st.error(
            "Local demo mode requires FL_ADMIN_PASSWORD to match the existing "
            "Admin account. Set it in the local process environment."
        )
        st.stop()
    local_admin = authenticate("admin", admin_password)
    if local_admin is None or local_admin.role != "admin":
        st.error(
            "Local demo sign-in failed. FL_ADMIN_PASSWORD must match the password "
            "already stored for the Admin account."
        )
        st.stop()
    st.session_state.authenticated_principal = local_admin
    if not st.session_state.get("_local_demo_home_initialized"):
        st.session_state.workspace_navigation = "Home"
        st.session_state._local_demo_home_initialized = True
elif "authenticated_principal" not in st.session_state:
    _, login_column, _ = st.columns([1, 1.35, 1])
    with login_column:
        with st.container(key="login-card"):
            st.markdown(
                """
                <div class="research-hero">
                    <h2>Research workspace</h2>
                    <p>Sign in to the Admin or simulated Hospital workspace.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.caption(
                "Local demonstration accounts only. Configure distinct account passwords "
                "before first launch. This prototype is not for production deployment."
            )
            with st.form("login_form"):
                username = st.text_input("Username", autocomplete="username")
                password = st.text_input(
                    "Password",
                    type="password",
                    autocomplete="current-password",
                )
                login_clicked = st.form_submit_button(
                    "Sign in",
                    type="primary",
                    width="stretch",
                )
    if login_clicked:
        principal = authenticate(username, password)
        if principal is None:
            st.error("Invalid username or password.")
        else:
            st.session_state.authenticated_principal = principal
            st.rerun()
    st.stop()

principal = st.session_state.authenticated_principal
if not isinstance(principal, Principal):
    del st.session_state.authenticated_principal
    st.rerun()

st.sidebar.success(f"Signed in: {principal.username} ({principal.role})")
if local_demo_enabled:
    st.sidebar.caption("Local demo mode · loopback connection only")

def navigate_to(page):
    if not page_is_allowed(principal.role, page):
        st.session_state.workspace_navigation = "Home"
        st.error("This page is not available to your account.")
        return
    st.session_state.workspace_navigation = page


if "workspace_navigation" not in st.session_state:
    st.session_state.workspace_navigation = "Home"
if not page_is_allowed(
    principal.role,
    st.session_state.workspace_navigation,
):
    st.session_state.workspace_navigation = "Home"

active_page = st.sidebar.radio(
    "Research workspace",
    pages_for_role(principal.role),
    key="workspace_navigation",
)
if not page_is_allowed(principal.role, active_page):
    st.error("This page is not available to your account.")
    st.stop()

if not local_demo_enabled and st.sidebar.button("Log out"):
    del st.session_state.authenticated_principal
    st.rerun()


def render_hospital_workspace(account, page):
    if account.role != "hospital" or not page_is_allowed(account.role, page):
        st.error("This page is not available to your account.")
        st.stop()

    st.markdown(
        """
        <div class="research-hero">
            <h2>Hospital workspace</h2>
            <p>Local dataset, training, prediction, and federated participation.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(f"Signed in as {account.username}")
    st.warning(
        "Research demonstration only. Do not upload identifiable patient data; "
        "predictions are not clinical diagnoses."
    )

    status = hospital_status(account)
    if page == "Home":
        st.subheader("Your hospital workspace")
        metrics = st.columns(4)
        metrics[0].metric("Local training records", status["training_records"])
        metrics[1].metric("Open rounds to join", status["open_rounds"])
        metrics[2].metric(
            "Completed participations",
            status["completed_participations"],
        )
        history = prediction_history(account)
        metrics[3].metric("Recorded predictions", len(history))
        st.info(
            "Your account is scoped to its assigned simulated hospital partition. "
            "You can review local data, participate in an Admin-opened round, "
            "generate research predictions, and view your prediction history."
        )
        st.subheader("Hospital pages")
        destinations = [
            ("Local Dataset", "Review this account's local partition summary."),
            ("Local Training", "See how local model training fits into a round."),
            ("Federated Participation", "Join an open round and submit one update."),
            ("Prediction", "Predict with the currently approved model."),
            ("Prediction History", "Review predictions recorded for this account."),
        ]
        for start in range(0, len(destinations), 3):
            columns = st.columns(3)
            for column, (destination, description) in zip(
                columns,
                destinations[start:start + 3],
            ):
                with column:
                    with st.container(border=True):
                        st.markdown(f"#### {destination}")
                        st.caption(description)
                        st.button(
                            f"Open {destination}",
                            key=f"hospital_home_{destination}",
                            width="stretch",
                            on_click=navigate_to,
                            args=(destination,),
                        )

    elif page == "Local Dataset":
        st.subheader("Local dataset summary")
        st.caption(
            "Counts below describe this account's assigned training partition; "
            "the held-out evaluation records are not part of local training."
        )
        columns = st.columns(3)
        columns[0].metric("Local training records", status["training_records"])
        columns[1].metric("Benign records", status["benign"])
        columns[2].metric("Malignant records", status["malignant"])
        st.info(
            "Hospital accounts can view only their own partition summary. "
            "The hospitals in this research prototype are simulated partitions, "
            "not separate institutions or independently secured servers."
        )

    elif page == "Local Training":
        st.subheader("Local model training")
        columns = st.columns(3)
        columns[0].metric("Records available for training", status["training_records"])
        columns[1].metric("Open rounds", status["open_rounds"])
        columns[2].metric(
            "Completed participations",
            status["completed_participations"],
        )
        st.write(
            "Local training is performed when this hospital submits its update "
            "to an Admin-opened federated round. Opening this page does not start "
            "training or change round state."
        )
        st.info(
            "To train and submit an update, open Federated Participation and select "
            "a round that is accepting this hospital's update."
        )
        st.button(
            "Open Federated Participation",
            key="hospital_local_training_participation",
            type="primary",
            width="stretch",
            on_click=navigate_to,
            args=("Federated Participation",),
        )

    elif page == "Federated Participation":
        st.subheader("Federated round participation")
        st.caption(
            "Training runs once for this hospital when you submit its update to "
            "an Admin-opened round. No new round is created from this page."
        )
        hospital_rounds = list_rounds(account)
        open_rounds = [
            item for item in hospital_rounds
            if item["status"] == "collecting" and not item["participated"]
        ]
        if open_rounds:
            selected_round = st.selectbox(
                "Open round",
                open_rounds,
                format_func=lambda item: f"{item['round_id']} — {item['mode']}",
                key="hospital_open_round",
            )
            if st.button(
                "Train locally and submit this hospital's update",
                type="primary",
                width="stretch",
                key="hospital_submit_update",
            ):
                try:
                    result = submit_hospital_update(account, selected_round["round_id"])
                    st.success(
                        f"Local update submitted ({result['sample_count']} training records); "
                        "raw feature rows were not included in the update."
                    )
                    st.rerun()
                except (PermissionError, ValueError, OSError) as error:
                    st.error(f"Local participation failed: {error}")
        elif not hospital_rounds:
            st.info("The administrator has not opened a federated round.")
        else:
            st.info("There are no open rounds awaiting your participation.")
        visible_rounds = [
            {
                "Round": item["round_id"],
                "Mode": item["mode"],
                "Status": item["status"],
                "Submitted updates": (
                    f"{item['participants']} / {item['expected_participants']}"
                ),
                "Aggregation": item["aggregation_status"],
                "Version deployed from round": (
                    item["global_model_version"] or "not deployed from this round"
                ),
                "Your participation": item["participated"],
            }
            for item in hospital_rounds
        ]
        if visible_rounds:
            st.dataframe(pd.DataFrame(visible_rounds), width="stretch", hide_index=True)
        with st.expander("DP-FedAvg: experimental privacy details"):
            details = privacy_description(account)
            st.write(f"**Server can observe:** {details['server_observes']}")
            st.warning(details["limitations"])

    elif page == "Prediction":
        st.subheader("Predict with the approved global model")
        model = current_model(account)
        if model:
            st.info(
                f"Predictions use only the approved model: "
                f"{model.get('version', model.get('name', 'approved model'))}."
            )
        else:
            st.error(
                "No model is approved for predictions yet. Contact the administrator."
            )
        uploaded = st.file_uploader(
            "Upload CSV with the 30 WDBC feature columns in canonical order",
            type=["csv"],
            key="hospital_prediction_csv",
        )
        if uploaded is not None:
            try:
                frame = pd.read_csv(uploaded)
                st.dataframe(frame.head(), width="stretch")
                if st.button("Predict uploaded records", type="primary"):
                    results = make_hospital_predictions(account, frame)
                    show_prediction_summary(results)
                    st.dataframe(results, width="stretch", hide_index=True)
                    st.download_button(
                        "Download predictions",
                        results.to_csv(index=False),
                        file_name="hospital_predictions.csv",
                        mime="text/csv",
                    )
            except (pd.errors.ParserError, UnicodeDecodeError, ValueError, TypeError, OSError) as error:
                st.error(f"Prediction failed: {error}")

        with st.expander("Enter one patient's features manually"):
            with st.form("manual_prediction_form"):
                feature_values = {}
                for start in range(0, len(FEATURES), 3):
                    columns = st.columns(3)
                    for column, feature in zip(columns, FEATURES[start:start + 3]):
                        feature_values[feature] = column.number_input(
                            feature, value=0.0, format="%.6f", key=f"manual_{feature}"
                        )
                manual_submit = st.form_submit_button(
                    "Predict this record",
                    type="primary",
                    width="stretch",
                )
            if manual_submit:
                try:
                    result = make_hospital_predictions(
                        account, pd.DataFrame([feature_values], columns=FEATURES)
                    )
                    show_prediction_summary(result)
                    st.dataframe(result, width="stretch", hide_index=True)
                except (ValueError, FileNotFoundError, OSError) as error:
                    st.error(f"Prediction failed: {error}")

    elif page == "Prediction History":
        st.subheader("Your prediction history")
        history = prediction_history(account)
        if history.empty:
            st.info("No predictions have been recorded for this hospital account.")
        else:
            st.dataframe(history, width="stretch", hide_index=True)


def show_prediction_summary(results):
    """Display readable result counts alongside the unchanged prediction table."""
    prediction_counts = results["Prediction"].value_counts()
    benign_count = int(prediction_counts.get("Benign", 0))
    malignant_count = int(prediction_counts.get("Malignant", 0))
    benign_col, malignant_col = st.columns(2)
    benign_col.metric("Predicted benign", benign_count)
    malignant_col.metric("Predicted malignant", malignant_count)
    st.caption(
        "Class labels use the existing 0.5 malignant-probability threshold. "
        "These model outputs are research predictions, not diagnoses."
    )


if principal.role == "hospital":
    render_hospital_workspace(principal, active_page)
    st.stop()

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_csv(filename):
    path = METRICS / filename

    if path.exists():
        return pd.read_csv(path)

    return pd.DataFrame()


LIVE_METRIC_LABELS = {
    "accuracy": "Accuracy (%)",
    "precision": "Precision (%)",
    "malignant_recall": "Malignant recall (%)",
    "specificity": "Specificity (%)",
    "f1_score": "F1 score (%)",
    "roc_auc": "ROC AUC (%)",
}


def format_metric_percentages_for_display(data):
    """Format a copy of measured fractional metrics as percentage values."""
    display = data.copy()
    for source, label in LIVE_METRIC_LABELS.items():
        if source in display.columns:
            display[label] = (
                pd.to_numeric(display[source], errors="raise") * 100
            ).round(2)
            display = display.drop(columns=[source])
    return display


# ============================================================
# LOAD EXISTING RESULTS
# ============================================================

comparison = load_csv(
    "final_three_model_comparison.csv"
)
comparison_artifacts = {
    "Centralized Logistic Regression": ["centralized_logistic_regression.pkl"],
    "FedAvg Logistic Regression": ["fedavg_model.npz", "fedavg_scaler.pkl"],
    "DP Federated Gradient Model": ["dp_federated_model.npz", "dp_federated_scaler.pkl"],
}
if not comparison.empty and "model" in comparison.columns:
    comparison["Saved artifact(s)"] = comparison["model"].map(
        lambda name: ", ".join(comparison_artifacts.get(name, []))
    )
    comparison["Artifact available"] = comparison["model"].map(
        lambda name: all(
            (MODELS / artifact).is_file()
            for artifact in comparison_artifacts.get(name, [])
        )
    )

four_comparison = load_csv(
    "four_model_comparison.csv"
)

fedavg_history = load_csv(
    "fedavg_training_history.csv"
)

dp_history = load_csv(
    "dp_federated_history.csv"
)

privacy_summary = load_csv(
    "privacy_repeated_summary.csv"
)

hospital_reliability = load_csv(
    "hospital_reliability.csv"
)

hospital_distribution = load_csv(
    "hospital_distribution.csv"
)

iid_comparison = load_csv(
    "iid_non_iid_comparison.csv"
)


# ============================================================
# HOME
# ============================================================

if active_page == "Home":
    st.markdown(
        """
        <div class="home-hero">
            <h1>Privacy-Preserving Federated Learning Framework for Breast Cancer Diagnosis</h1>
            <p>
                This academic study compares centralized and federated
                Logistic Regression on the Wisconsin Diagnostic Breast Cancer
                dataset, examining whether model updates can be combined across
                simulated hospital partitions without sharing their raw rows.
                DP-FedAvg is experimental; end-to-end differential privacy is
                not established.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.warning(
        "Academic research prototype—not for clinical diagnosis. Hospital clients "
        "are simulated partitions running in one application process, not separate "
        "medical institutions or independently secured servers."
    )

    dataset_path = BASE / "data" / "wdbc.data"
    if dataset_path.is_file():
        dataset = pd.read_csv(dataset_path, header=None)
        total_records = len(dataset)
        stats = st.columns(3)
        stats[0].metric("WDBC records", f"{total_records:,}")
        stats[1].metric("Input features", len(FEATURES))
        stats[2].metric(
            "Simulated hospitals",
            len(hospital_distribution) if not hospital_distribution.empty else "Unavailable",
        )
    else:
        st.error(f"Dataset not found: {dataset_path}")

    active_model = current_model(principal)
    st.subheader("Deployment status")
    if active_model:
        model_columns = st.columns([1, 2])
        model_columns[0].metric("Approved model", active_model.get("name", "Approved model"))
        model_columns[1].metric(
            "Active version",
            active_model.get("version", "Not versioned"),
        )
    else:
        st.warning(
            "No model is currently approved. Hospital predictions remain unavailable "
            "until an administrator explicitly approves a model."
        )

    st.subheader("Simulated hospital partitions")
    if not hospital_distribution.empty:
        hospital_cards = st.columns(min(3, len(hospital_distribution)))
        for column, (_, hospital) in zip(
            hospital_cards,
            hospital_distribution.iterrows(),
        ):
            with column:
                total = int(hospital["total_patients"])
                with st.container(border=True):
                    st.markdown(f"#### {hospital['hospital']}")
                    st.metric("Training records", f"{total:,}")
                    st.caption(
                        f"Benign {int(hospital['benign_patients']):,} · "
                        f"Malignant {int(hospital['malignant_patients']):,}"
                    )
    else:
        st.info("Hospital partition statistics are unavailable.")

    st.subheader("Compared approaches")
    approach_columns = st.columns(3)
    approaches = [
        (
            "Centralized Logistic Regression",
            "Reference model trained centrally on the development partition.",
        ),
        (
            "FedAvg",
            "Simulated hospital model updates combined using sample-count weights.",
        ),
        (
            "Experimental DP-FedAvg",
            "Client-side clipping and noisy updates; accounting is conditional and "
            "does not establish end-to-end differential privacy.",
        ),
    ]
    for column, (name, description) in zip(approach_columns, approaches):
        with column:
            with st.container(border=True):
                st.markdown(f"#### {name}")
                st.caption(description)

    st.subheader("Model comparison")
    st.caption(
        "Historical held-out results already stored with this project. No training "
        "or experiment is started by opening this page. The saved DP Federated "
        "Gradient Model is a separate historical approach, not live DP-FedAvg."
    )
    if not comparison.empty:
        home_comparison = format_comparison_for_display(comparison)
        home_columns = [
            "model",
            "Accuracy (%)",
            "Precision (%)",
            "Malignant Recall (%)",
            "F1 Score (%)",
            "ROC AUC (%)",
        ]
        home_columns = [
            name for name in home_columns if name in home_comparison.columns
        ]
        st.dataframe(
            home_comparison.loc[:, home_columns],
            width="stretch",
            hide_index=True,
            column_config={
                "model": st.column_config.TextColumn("Model", width="medium"),
            },
        )
    else:
        st.info("Historical model comparison results are unavailable.")
    st.button(
        "Open full model comparison",
        key="home_open_model_comparison",
        type="primary",
        width="stretch",
        on_click=navigate_to,
        args=("Models",),
    )

    st.subheader("Explore the research workspace")
    navigation_cards = [
        ("Overview", "Dataset distribution and the federated workflow."),
        ("Hospitals", "View simulated client partitions and historical results."),
        ("Models", "Compare stored results and run an explicitly requested evaluation."),
        ("Privacy", "Review conditional DP accounting and its limitations."),
        ("Prediction", "Generate research predictions with the approved model."),
        ("Retraining", "Register verified labels and review candidate models."),
        ("Federated Operations", "Start a round, collect updates, and review candidates."),
        ("Experimental Evaluation", "Run held-out-safe validation experiments."),
    ]
    for start in range(0, len(navigation_cards), 3):
        columns = st.columns(3)
        for column, (page, description) in zip(
            columns,
            navigation_cards[start:start + 3],
        ):
            with column:
                with st.container(border=True):
                    st.markdown(f"#### {page}")
                    st.caption(description)
                    st.button(
                        f"Open {page}",
                        key=f"home_navigate_{page}",
                        width="stretch",
                        on_click=navigate_to,
                        args=(page,),
                    )


# ============================================================
# OVERVIEW
# ============================================================

if active_page == "Overview":

    st.header("Project Overview")
    st.caption(
        "Explore the dataset, model lifecycle, and current deployment state. "
        "Live evaluation runs only when explicitly started."
    )

    c1, c2, c3, c4 = st.columns(4)

    dataset_path = BASE / "data" / "wdbc.data"
    if dataset_path.exists():
        dataset = pd.read_csv(dataset_path, header=None)
        total_patients = len(dataset)
        training_patients = (
            int(hospital_distribution["total_patients"].sum())
            if "total_patients" in hospital_distribution.columns
            else 0
        )
        testing_patients = total_patients - training_patients
        c1.metric("Total records", total_patients)
        c2.metric("Train records", training_patients)
        c3.metric("Test records", testing_patients)
    else:
        st.error(f"Dataset not found: {dataset_path}")
    c4.metric(
        "Simulated Hospitals",
        len(hospital_distribution)
        if not hospital_distribution.empty
        else "Unavailable",
    )

    active_model = current_model(principal)
    if active_model:
        st.success(
            f"Approved model: {active_model.get('name', 'Approved model')} · "
            f"version {active_model.get('version', 'not versioned')}"
        )
    else:
        st.warning(
            "No approved model is active. Predictions are unavailable until an "
            "administrator explicitly approves a model."
        )

    st.subheader("Project Workflow")

    workflow_steps = [
        (
            "01 · WDBC data",
            "Local Wisconsin Diagnostic Breast Cancer dataset with 30 input features.",
        ),
        (
            "02 · Three client partitions",
            "Disjoint simulated hospital partitions; this is not a multi-server deployment.",
        ),
        (
            "03 · Local Logistic Regression",
            "Each simulated hospital trains on its own assigned training records.",
        ),
        (
            "04 · FedAvg aggregation",
            "The coordinator combines submitted parameters by each client's sample count.",
        ),
        (
            "05 · Candidate review",
            "An aggregate remains pending until an administrator explicitly approves it.",
        ),
        (
            "06 · Evaluation and prediction",
            "Compare actual model predictions; hospital inference uses the approved model.",
        ),
    ]
    for start in (0, 3):
        step_columns = st.columns(3)
        for column, (title, description) in zip(
            step_columns,
            workflow_steps[start:start + 3],
        ):
            with column:
                st.markdown(
                    f"""
                    <div class="workflow-step">
                        <strong>{title}</strong>
                        <span>{description}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    st.subheader("Overall Dataset Distribution")

    if dataset_path.exists():
        dataset_distribution = pd.DataFrame({
            "Diagnosis": ["Benign", "Malignant"],
            "Patients": [
                int((dataset.iloc[:, 1] == "B").sum()),
                int((dataset.iloc[:, 1] == "M").sum()),
            ],
        })
        st.altair_chart(
            horizontal_metric_chart(
                dataset_distribution,
                category="Diagnosis",
                metric="Patients",
                title="Dataset diagnosis distribution",
            ),
            width="stretch",
        )
        st.caption("Counts are read from the local WDBC dataset.")


# ============================================================
# HOSPITALS
# ============================================================

if active_page == "Hospitals":

    st.header("Three Simulated Hospitals")

    hospital_df = hospital_distribution.rename(columns={
        "hospital": "Hospital",
        "benign_patients": "Benign",
        "malignant_patients": "Malignant",
    })
    if not hospital_df.empty:
        hospital_df["Total"] = hospital_df["Benign"] + hospital_df["Malignant"]
        hospital_df["Malignant Percentage"] = (
            hospital_df["Malignant"] / hospital_df["Total"] * 100
        ).round(2)
        participation_by_partition = {
            (item["benign"], item["malignant"]): item
            for item in all_hospital_statuses(principal)
        }

        st.subheader("Hospital workspaces")
        hospital_columns = st.columns(3)
        for column, (_, hospital) in zip(
            hospital_columns,
            hospital_df.iterrows(),
        ):
            with column:
                hospital_name = hospital["Hospital"]
                with st.container(border=True):
                    st.markdown(f"#### {hospital_name}")
                    st.metric("Training records", int(hospital["Total"]))
                    st.caption(f"Benign: {int(hospital['Benign']):,}")
                    st.caption(f"Malignant: {int(hospital['Malignant']):,}")
                    st.caption(
                        f"{hospital['Malignant Percentage']:.1f}% malignant in this partition"
                    )
                    participation = participation_by_partition.get(
                        (int(hospital["Benign"]), int(hospital["Malignant"]))
                    )
                    if participation:
                        st.caption(
                            f"Participation: {participation['completed_participations']} complete · "
                            f"{participation['open_rounds']} open"
                        )
                    else:
                        st.caption("Participation status unavailable.")

        st.subheader("Partition distribution")
        st.altair_chart(
            horizontal_grouped_count_chart(
                hospital_df,
                category="Hospital",
                metrics=["Benign", "Malignant"],
                title="Patients by Simulated Hospital",
            ),
            width="stretch",
        )
    else:
        st.warning("Hospital partition results are not available.")

    st.info(
        "The hospitals are simulated partitions, "
        "not real medical institutions. "
        "Their different malignant proportions "
        "represent non-IID data."
    )

    if not hospital_reliability.empty:

        with st.expander("Historical hospital reliability results"):
            st.dataframe(
                hospital_reliability,
                width="stretch",
                hide_index=True,
            )

    if not iid_comparison.empty:
        with st.expander("Stored IID / Non-IID comparison"):
            st.dataframe(
                iid_comparison,
                width="stretch",
                hide_index=True,
            )


# ============================================================
# MODEL COMPARISON
# ============================================================

if active_page == "Models":

    st.header("Model Performance Comparison")

    st.subheader("Live held-out Logistic Regression comparison")
    st.caption(
        "Runs centralized Logistic Regression, sample-count-weighted FedAvg, and "
        "experimental DP-FedAvg against the same stratified 114-record WDBC holdout. "
        "The scaler is fit on the 455 training records only. These exploratory test "
        "results must not be used to tune or select a deployment model."
    )
    if st.button(
        "Run live model comparison",
        type="primary",
        key="run_live_model_comparison",
    ):
        try:
            with st.spinner("Training and evaluating the three Logistic Regression approaches..."):
                live_results, live_metadata = run_live_model_comparison()
            st.session_state.live_model_comparison = live_results
            st.session_state.live_model_comparison_metadata = live_metadata
        except (ValueError, OSError, RuntimeError) as error:
            st.error(f"Live model comparison failed: {error}")

    live_results = st.session_state.get("live_model_comparison")
    if live_results is not None:
        st.caption("Newly measured in this dashboard session; not a saved historical result.")
        live_display = format_metric_percentages_for_display(live_results)
        setup_columns = [
            "approach",
            "partition",
            "training_records",
            "validation_records",
            "training_time_seconds",
        ]
        score_columns = [
            LIVE_METRIC_LABELS[name]
            for name in (
                "accuracy",
                "precision",
                "malignant_recall",
                "f1_score",
                "roc_auc",
            )
            if LIVE_METRIC_LABELS[name] in live_display.columns
        ]
        st.markdown("**Evaluation setup**")
        st.dataframe(
            live_display.loc[
                :,
                [name for name in setup_columns if name in live_display.columns],
            ],
            width="stretch",
            hide_index=True,
            column_config={
                "approach": st.column_config.TextColumn("Model", width="medium"),
                "partition": st.column_config.TextColumn("Partition", width="small"),
                "training_records": st.column_config.NumberColumn(
                    "Training records",
                    format="%d",
                    width="small",
                ),
                "validation_records": st.column_config.NumberColumn(
                    "Held-out records",
                    format="%d",
                    width="small",
                ),
                "training_time_seconds": st.column_config.NumberColumn(
                    "Fit time (s)",
                    format="%.2f",
                    width="small",
                ),
            },
        )
        st.markdown("**Measured performance**")
        st.dataframe(
            live_display.loc[:, score_columns],
            width="stretch",
            hide_index=True,
            column_config={
                label: st.column_config.NumberColumn(
                    label,
                    format="%.2f",
                    width="small",
                )
                for label in score_columns
            },
        )
        live_metadata = st.session_state.live_model_comparison_metadata
        st.write(
            "Hospital training-partition sizes:",
            ", ".join(
                f"{item['hospital']}: {item['records']}"
                for item in live_metadata["hospital_partitions"]
            ),
        )
        st.caption(
            f"Training setup: {live_metadata['rounds']} aggregation rounds × "
            f"{live_metadata['local_epochs']} local epochs; "
            f"{live_metadata['training_records']} training and "
            f"{live_metadata['held_out_records']} held-out records."
        )

    if comparison.empty:

        st.warning(
            "Model comparison results were not found."
        )

    else:

        st.subheader("Final Test Results")

        display_comparison = format_comparison_for_display(comparison)
        comparison_columns = [
            "model",
            "Accuracy (%)",
            "Precision (%)",
            "Malignant Recall (%)",
            "F1 Score (%)",
            "ROC AUC (%)",
        ]
        compact_comparison = display_comparison.loc[
            :,
            [name for name in comparison_columns if name in display_comparison],
        ]
        st.dataframe(
            compact_comparison,
            width="stretch",
            hide_index=True,
            column_config={
                "model": st.column_config.TextColumn("Model", width="medium"),
                "Malignant Recall (%)": st.column_config.NumberColumn(
                    "Malignant recall",
                    format="%.2f%%",
                    width="small",
                ),
            },
        )
        st.caption(
            "Results are loaded from final_three_model_comparison.csv. "
            "Percentages are display-formatted; source result values are unchanged."
        )
        if "Artifact available" in display_comparison and not display_comparison["Artifact available"].all():
            st.error("One or more comparison rows do not have all expected model artifacts.")

        recall_chart_col, false_negative_chart_col = st.columns(2)
        if "model" in comparison and "malignant_recall" in comparison:
            with recall_chart_col:
                st.altair_chart(
                    horizontal_metric_chart(
                        comparison,
                        category="model",
                        metric="malignant_recall",
                        title="Malignant recall",
                        percentage=True,
                    ),
                    width="stretch",
                )
        if "model" in comparison and "false_negatives" in comparison:
            with false_negative_chart_col:
                st.altair_chart(
                    horizontal_metric_chart(
                        comparison,
                        category="model",
                        metric="false_negatives",
                        title="False negatives (patients)",
                    ),
                    width="stretch",
                )

        st.subheader("Model approval")
        comparable_models = [
            name for name in MODEL_ARTIFACTS
            if name in comparison["model"].astype(str).tolist()
        ]
        active_descriptor = (
            json.loads(ACTIVE_MODEL_PATH.read_text(encoding="utf-8"))
            if ACTIVE_MODEL_PATH.is_file()
            else None
        )
        if active_descriptor:
            st.info(f"Currently approved for inference: {active_descriptor.get('name', 'Unknown model')}")
        else:
            st.info("No model has been approved for inference yet.")
        if comparable_models:
            default_index = (
                comparable_models.index(active_descriptor["name"])
                if active_descriptor and active_descriptor.get("name") in comparable_models
                else 0
            )
            selected_model = st.selectbox(
                "Model to inspect",
                comparable_models,
                index=default_index,
                key="comparison_model_selection",
            )
            selected_result = comparison.loc[
                comparison["model"].astype(str) == selected_model
            ]
            with st.expander("Inspect full saved metrics and artifacts"):
                st.dataframe(
                    format_comparison_for_display(selected_result),
                    width="stretch",
                    hide_index=True,
                )
            selected_artifact = MODEL_ARTIFACTS[selected_model]
            selected_available = (
                (MODELS / selected_artifact["model"]).is_file()
                and (
                    selected_artifact["kind"] == "pipeline"
                    or (MODELS / selected_artifact["scaler"]).is_file()
                )
            )
            if not selected_available:
                st.error("This model or its required scaler is unavailable.")
            approval_confirmed = st.checkbox(
                f"I explicitly approve {selected_model} for inference.",
                key=f"baseline_model_approval_confirmation_{selected_model}",
            )
            if st.button(
                "Approve selected model for inference",
                type="primary",
                disabled=not approval_confirmed or not selected_available,
                key="approve_selected_baseline_model",
            ):
                try:
                    active = approve_baseline_model(
                        selected_model,
                        admin_approved=approval_confirmed,
                        principal=principal,
                    )
                    st.success(f"{active['name']} is now approved for inference.")
                    st.rerun()
                except (ValueError, OSError, FileNotFoundError, PermissionError) as error:
                    st.error(f"Model approval failed: {error}")

        if {"model", "accuracy"}.issubset(
            comparison.columns
        ):

            accuracy_df = comparison[
                ["model", "accuracy"]
            ].copy()

            st.subheader("Accuracy Comparison (%)")

            st.altair_chart(
                horizontal_metric_chart(
                    accuracy_df,
                    category="model",
                    metric="accuracy",
                    title="Accuracy",
                    percentage=True,
                ),
                width="stretch",
            )

    st.subheader("FedAvg Training Progress")

    if not fedavg_history.empty:

        history_chart = fedavg_training_history_chart(fedavg_history)
        if history_chart is not None:
            st.altair_chart(history_chart, width="stretch")
        else:
            st.info("No finite FedAvg loss observations are available to chart.")

        with st.expander("View FedAvg Training History"):

            st.dataframe(
                fedavg_history,
                width="stretch",
                hide_index=True
            )

    else:

        st.warning(
            "FedAvg training history not found."
        )

    with st.expander(
        "Fixed-Scaler DP Experiment"
    ):

        if not four_comparison.empty:

            st.dataframe(
                four_comparison,
                width="stretch",
                hide_index=True
            )

            st.warning(
                "This is a separate exploratory result, not the selected "
                "DP-FedAvg artifact used in the three-model comparison. "
                "Review its saved metrics before interpreting its performance."
            )

        else:

            st.write(
                "Additional experiment results unavailable."
            )


# ============================================================
# DIFFERENTIAL PRIVACY
# ============================================================

if active_page == "Privacy":

    st.header("Differential Privacy Analysis")
    st.caption(
        "Dashboard DP-FedAvg is experimental. Values shown are conditional mechanism "
        "accounting, not a deployment-wide privacy guarantee."
    )

    st.subheader("Mechanism and release boundary")
    privacy = privacy_description(principal)
    st.write(f"**Randomness:** {privacy['noise_randomness']}")
    st.write(f"**Coordinator visibility:** {privacy['server_observes']}")
    st.warning(privacy["limitations"])
    with st.expander("How the experimental DP-FedAvg mechanism works"):
        st.markdown("""
        Each simulated client computes per-record Logistic Regression
        weight-and-bias gradients, clips each concatenated gradient to L2 norm
        `C`, then adds Gaussian noise to each full-batch mean. For a fixed-size
        hospital partition of `n` records, the implementation uses standard
        deviation `noise_multiplier × 2C/n` under replace-one-record adjacency.
        The client trains for the configured local epochs and submits only its
        final noisy model parameters; the coordinator then applies ordinary
        sample-count-weighted FedAvg.

        Accounting uses the Gaussian mechanism's RDP bound, composed sequentially
        over every submitted client update and local epoch, with no sampling
        amplification. The displayed epsilon/delta is conditional on ideal
        Gaussian noise and the stated fixed-size adjacency, partitions, and
        counts. In this single-process demo, the noise-before-submission boundary
        is not enforced against coordinator access. This is not an end-to-end
        privacy guarantee.

        Smaller epsilon indicates a tighter conditional privacy bound; delta is
        the approximate-DP failure probability parameter. Neither value covers
        the complete application release surface.
        """)

    live_dp_results = st.session_state.get("live_model_comparison")
    if live_dp_results is not None:
        live_dp_results = live_dp_results.loc[
            live_dp_results["noise_multiplier"].notna(),
            [
                "partition",
                "noise_multiplier",
                "clipping_norm",
                "rounds",
                "local_epochs",
                "accounted_client_epoch_mechanisms",
                "epsilon_upper_bound",
                "delta",
                "accuracy",
                "precision",
                "malignant_recall",
                "f1_score",
                "roc_auc",
            ],
        ]
        st.subheader("Live privacy–utility comparison")
        st.caption(
            "Rows are measured during the explicitly requested comparison, using the "
            "same held-out records. These metrics are not included in the privacy bound."
        )
        utility_chart_data = live_dp_results.copy()
        utility_chart_data["DP setting"] = utility_chart_data.apply(
            lambda row: (
                f"{row['partition']} · noise σ={row['noise_multiplier']:g}"
            ),
            axis=1,
        )
        st.altair_chart(
            horizontal_metric_chart(
                utility_chart_data,
                category="DP setting",
                metric="accuracy",
                title="Measured accuracy across DP noise settings",
                percentage=True,
            ),
            width="stretch",
        )
        st.dataframe(
            format_metric_percentages_for_display(live_dp_results),
            width="stretch",
            hide_index=True,
        )
        st.caption(
            "These epsilon/delta values bound only the modeled noisy client-update "
            "mechanism under the assumptions above; model metrics and other app outputs "
            "are not included."
        )

    accounting = dp_privacy_accounting(principal)
    if accounting:
        st.subheader("Cumulative accountant output")
        accounting_cols = st.columns(4)
        accounting_cols[0].metric(
            "Conditional epsilon",
            f"≤ {accounting['epsilon_upper_bound']:.4f}",
        )
        accounting_cols[1].metric("Delta", f"{accounting['delta']:g}")
        accounting_cols[2].metric(
            "Client/epoch mechanisms",
            accounting["accounted_client_epoch_mechanisms"],
        )
        accounting_cols[3].metric(
            "Client updates",
            accounting["accounted_client_updates"],
        )
        st.caption(
            "Cumulative composition includes every persisted DP update, including "
            "partial or cancelled rounds. Delta is taken from the most recently "
            "opened DP round."
        )
        st.caption(
            f"Accountant: {accounting['accountant_method']}; "
            f"adjacency: {accounting['adjacency']}; "
            f"sampling assumption: {accounting['sampling']}."
        )
        if accounting["accounted_rounds"]:
            latest_round = accounting["accounted_rounds"][-1]
            st.subheader("Most recent submitted-round parameters")
            parameter_cols = st.columns(3)
            parameter_cols[0].metric(
                "Per-record clip norm",
                latest_round["clipping_norm"],
            )
            parameter_cols[1].metric(
                "Gaussian noise multiplier",
                latest_round["noise_multiplier"],
            )
            parameter_cols[2].metric(
                "Local epochs",
                latest_round["local_epochs"],
            )
            st.dataframe(
                pd.DataFrame(accounting["accounted_rounds"]),
                width="stretch",
                hide_index=True,
            )
        st.caption(accounting["accounting_scope"])
    else:
        st.info("No DP-FedAvg round has been opened; there are no submitted DP updates to account for.")

    st.subheader("Historical experiments")
    c1, c2 = st.columns(2)
    c1.metric("Historical DP log rows", len(dp_history))
    c2.metric("Historical privacy-utility rows", len(privacy_summary))

    if not dp_history.empty:

        st.subheader("Historical DP training output (separate experiment)")
        st.caption(
            "This saved history is not the dashboard's noisy client-update execution path."
        )

        with st.expander("View DP Training History"):

            st.dataframe(
                dp_history,
                width="stretch",
                hide_index=True
            )

    if not privacy_summary.empty:
        with st.expander("View saved privacy-utility experiment results"):
            st.dataframe(
                privacy_summary,
                width="stretch",
                hide_index=True,
            )

    st.warning(
        "These saved DP experiment results are historical and come from the "
        "older coordinator path, which receives per-hospital clipped gradient "
        "sums before noise addition. They are not the DP-FedAvg client-side "
        "workflow offered under Federated Operations and must not be treated "
        "as its metrics. No end-to-end differential privacy guarantee is claimed."
    )


# ============================================================
# TAB 5: PREDICTION
# ============================================================

if active_page == "Prediction":

    st.header("Interactive Breast Cancer Prediction")
    st.caption(
        f"Upload a CSV with {len(FEATURES)} WDBC features in canonical order. "
        "Batch predictions use the currently approved model."
    )
    st.warning(
        "For academic demonstration only. "
        "Do not upload identifiable patient records. "
        "This is not a clinical diagnostic system."
    )

    input_column, model_column = st.columns([1.3, 1])
    with input_column:
        uploaded_file = st.file_uploader(
            "Upload prediction batch (CSV)",
            type=["csv"],
            help="Include all 30 numeric WDBC feature columns in canonical order.",
        )
    with model_column:
        if ACTIVE_MODEL_PATH.is_file():
            active_descriptor = json.loads(ACTIVE_MODEL_PATH.read_text(encoding="utf-8"))
            model_choice = active_descriptor.get("name", "Approved model")
            st.markdown("#### Active inference model")
            st.success(
                f"Approved · {active_descriptor.get('version', 'current')}"
            )
            st.caption(model_choice)
        else:
            model_choice = None
            st.warning(
                "No model is approved. An administrator must approve a model "
                "before predictions can run."
            )

    if uploaded_file is not None:
        try:
            incoming = pd.read_csv(uploaded_file)
            features = validate_feature_frame(incoming)
            st.subheader("Validated input")
            st.dataframe(
                features.head(),
                width="stretch",
                hide_index=True,
            )
            st.caption(
                f"Validated {len(features)} records with {len(FEATURES)} features "
                "in the canonical WDBC feature order."
            )

            if model_choice and st.button("Generate Predictions", type="primary"):
                probabilities = predict_probabilities(
                    features,
                    active_path=ACTIVE_MODEL_PATH,
                )
                predictions = np.where(probabilities >= 0.5, "Malignant", "Benign")
                results = pd.DataFrame({
                    "Record": np.arange(1, len(predictions) + 1),
                    "Benign Probability": 1.0 - probabilities,
                    "Malignant Probability": probabilities,
                    "Prediction": predictions,
                    "Model": model_choice,
                })
                st.success(f"Predictions generated with {model_choice}.")
                show_prediction_summary(results)
                st.dataframe(
                    results,
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "Record": st.column_config.NumberColumn(format="%d"),
                        "Benign Probability": st.column_config.NumberColumn(
                            "Benign probability",
                            format="percent",
                        ),
                        "Malignant Probability": st.column_config.NumberColumn(
                            "Malignant probability",
                            format="percent",
                        ),
                        "Prediction": st.column_config.TextColumn("Prediction"),
                        "Model": st.column_config.TextColumn("Model", width="medium"),
                    },
                )
                st.download_button(
                    "Download Predictions",
                    data=results.to_csv(index=False),
                    file_name="demo_predictions.csv",
                    mime="text/csv",
                )
        except (pd.errors.ParserError, UnicodeDecodeError, ValueError, TypeError) as error:
            st.error(f"Unable to validate uploaded file: {error}")
        except (OSError, KeyError) as error:
            st.error(f"Unable to load or use the selected model: {error}")


# ============================================================
# TAB 6: CONTROLLED RETRAINING AND APPROVAL
# ============================================================

if active_page == "Retraining":
    st.header("Admin: Confirmed-Label Retraining")
    st.warning(
        "Admin-only prototype control. Predictions are never labels; "
        "only newly verified ground truth can be registered."
    )
    setup_column, approval_column = st.columns([1, 1.15])
    with setup_column:
        with st.container(border=True):
            st.subheader("Data and training controls")
            pending = pending_confirmed_records(principal)
            st.metric("Verified records awaiting training", len(pending))
            threshold = st.number_input(
                "Minimum records before training",
                min_value=6,
                value=100,
                step=1,
                help="Configurable demonstration threshold, not a validated clinical requirement.",
            )
            labels_file = st.file_uploader(
                "Upload confirmed labels (30 features plus diagnosis)",
                type=["csv"],
                key="confirmed_labels_upload",
            )
            confirm_labels = st.checkbox(
                "I confirm all uploaded diagnoses are newly verified ground truth."
            )
            if labels_file is not None and st.button(
                "Register confirmed labels",
                key="register_confirmed_labels",
            ):
                try:
                    confirmed_frame = pd.read_csv(labels_file)
                    added = register_confirmed_records(
                        confirmed_frame,
                        admin_confirmed=confirm_labels,
                        principal=principal,
                    )
                    st.success(f"Registered {added} newly confirmed labeled records.")
                    st.rerun()
                except (pd.errors.ParserError, UnicodeDecodeError, ValueError, TypeError, PermissionError) as error:
                    st.error(f"Confirmed records were not registered: {error}")
                except OSError as error:
                    st.error(f"Unable to persist confirmed records: {error}")

            if st.button(
                "Train and evaluate candidate",
                type="primary",
                disabled=len(pending) < threshold,
            ):
                try:
                    manifest = create_candidate(
                        minimum_records=int(threshold),
                        principal=principal,
                    )
                    st.success(
                        f"Candidate {manifest['candidate_id']} trained and evaluated."
                    )
                    with st.expander("Candidate evaluation details"):
                        st.json(manifest)
                    st.rerun()
                except (ValueError, OSError, FileNotFoundError) as error:
                    st.error(f"Candidate training failed: {error}")

    with approval_column:
        with st.container(border=True):
            st.subheader("Candidate review and approval")
            candidate_files = (
                sorted(CANDIDATE_DIR.glob("*.json"))
                if CANDIDATE_DIR.exists()
                else []
            )
            pending_candidates = []
            for candidate_file in candidate_files:
                candidate_manifest = json.loads(
                    candidate_file.read_text(encoding="utf-8")
                )
                if candidate_manifest.get("status") == "pending_approval":
                    pending_candidates.append(candidate_manifest)

            if pending_candidates:
                selected_candidate = st.selectbox(
                    "Pending candidate",
                    pending_candidates,
                    format_func=lambda item: item["candidate_id"],
                )
                st.caption(
                    f"Model: {selected_candidate.get('name', 'Candidate')} · "
                    f"status: {selected_candidate.get('status', 'unknown')}"
                )
                with st.expander("Inspect candidate manifest"):
                    st.json(selected_candidate)
                approve_confirmed = st.checkbox(
                    f"I explicitly approve replacing the active model with "
                    f"{selected_candidate['candidate_id']}.",
                    key=f"candidate_approval_confirmation_{selected_candidate['candidate_id']}",
                )
                if st.button(
                    "Approve candidate and activate",
                    type="primary",
                    disabled=not approve_confirmed,
                ):
                    try:
                        active = approve_candidate(
                            selected_candidate["candidate_id"],
                            admin_approved=approve_confirmed,
                            principal=principal,
                        )
                        st.success(f"Active model updated to {active['name']}.")
                        st.rerun()
                    except (ValueError, OSError, FileNotFoundError, PermissionError) as error:
                        st.error(f"Candidate approval failed: {error}")
            else:
                st.info("No evaluated candidates are awaiting approval.")

            if ACTIVE_MODEL_PATH.exists():
                st.caption(f"Active-model pointer: {ACTIVE_MODEL_PATH.name}")


# ============================================================
# TAB 7: AUTHENTICATED FEDERATED OPERATIONS
# ============================================================

if active_page == "Federated Operations":
    st.header("Federated round administration")
    st.caption(
        "Three in-process simulated hospital clients train on disjoint fixed training "
        "partitions and submit model parameters, not raw training rows. The coordinator "
        "applies sample-count-weighted FedAvg. Candidate artifacts remain separate "
        "from the active model until explicit Admin approval."
    )
    st.warning(
        "This is a three-hospital simulation in one application process, not three "
        "independently secured hospital servers."
    )

    workflow_columns = st.columns(4)
    for column, (title, description) in zip(
        workflow_columns,
        [
            ("1 · Local training", "Each hospital trains on its fixed local partition."),
            ("2 · Update submission", "Only model parameters and sample count are submitted."),
            ("3 · FedAvg aggregation", "Parameters are weighted by each hospital's sample count."),
            ("4 · Candidate review", "Admin approval is required before deployment."),
        ],
    ):
        with column:
            st.markdown(
                f"""
                <div class="workflow-step">
                    <strong>{title}</strong>
                    <span>{description}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
    st.subheader("Hospital status")
    try:
        statuses = all_hospital_statuses(principal)
        st.dataframe(pd.DataFrame(statuses), width="stretch", hide_index=True)
    except (PermissionError, OSError, ValueError) as error:
        st.error(f"Unable to load hospital status: {error}")

    active_model = current_model(principal)
    st.caption(
        "Currently approved global model version: "
        + (
            active_model.get("version", active_model.get("name", "unknown"))
            if active_model
            else "none"
        )
    )

    st.subheader("Start a federated round")
    training_mode = st.selectbox(
        "Aggregation mode",
        ["FedAvg", "DP-FedAvg (experimental)"],
        help="DP-FedAvg applies per-record gradient clipping and client-side Gaussian noise; it is not a formal DP guarantee.",
    )
    epochs = st.number_input("Local epochs per hospital", min_value=1, max_value=50, value=5)
    dp_options = {}
    if training_mode.startswith("DP"):
        dp_col1, dp_col2, dp_col3 = st.columns(3)
        clipping = dp_col1.number_input("Per-record gradient clip norm", min_value=0.01, value=1.0)
        noise = dp_col2.number_input("Gaussian noise multiplier", min_value=0.1, value=2.0)
        delta = dp_col3.number_input(
            "Accounting delta (conditional bound)",
            min_value=1e-8,
            max_value=0.1,
            value=1e-5,
            format="%.8f",
        )
        dp_options = {
            "clipping_norm": clipping,
            "noise_multiplier": noise,
            "delta": delta,
        }
        provisional = DPAccountant(
            noise_multiplier=noise,
            delta=delta,
            sensitivity=1.0,
        )
        provisional.record_steps(3 * int(epochs))
        st.info(
            f"Conditional one-round Gaussian/RDP mechanism bound: epsilon ≤ "
            f"{provisional.get_epsilon():.4f}, delta={delta:g}, "
            f"{provisional.steps} client/epoch mechanisms (3 clients × "
            f"{int(epochs)} local epochs). This preview excludes prior rounds, "
            "pooled preprocessing, metrics, and other system-level leakage; it "
            "is not an end-to-end privacy guarantee."
        )

    if st.button("Open round for hospital participation", type="primary"):
        try:
            round_id = start_federated_round(
                principal,
                mode=training_mode,
                local_epochs=int(epochs),
                **dp_options,
            )
            st.success(
                f"Round {round_id} opened. Sign in separately as each hospital "
                "and submit its local update."
            )
            st.rerun()
        except (PermissionError, ValueError, OSError) as error:
            st.error(f"Unable to start federated round: {error}")

    st.subheader("Round progress and candidate review")
    try:
        rounds = list_rounds(principal)
    except (PermissionError, OSError, ValueError) as error:
        rounds = []
        st.error(f"Unable to load round progress: {error}")
    if rounds:
        progress_rows = []
        for item in rounds:
            progress_rows.append({
                "Round": item["round_id"],
                "Mode": item["mode"],
                "Status": item["status"].replace("_", " ").title(),
                "Submitted updates": (
                    f"{item['participants']} / {item['expected_participants']}"
                ),
                "Aggregation": item["aggregation_status"].replace("_", " ").title(),
                "Model version": item["global_model_version"] or "—",
                "Candidate": (
                    item["candidate"]["candidate_id"]
                    if item.get("candidate") else ""
                ),
            })
        st.dataframe(
            pd.DataFrame(progress_rows),
            width="stretch",
            hide_index=True,
            column_config={
                "Round": st.column_config.TextColumn(width="small"),
                "Mode": st.column_config.TextColumn(width="medium"),
                "Status": st.column_config.TextColumn(width="small"),
                "Submitted updates": st.column_config.TextColumn(width="small"),
                "Aggregation": st.column_config.TextColumn(width="medium"),
                "Model version": st.column_config.TextColumn(width="small"),
                "Candidate": st.column_config.TextColumn(width="small"),
            },
        )
        for item in rounds:
            if item.get("participant_results"):
                with st.expander(f"Client results — {item['round_id']}"):
                    st.dataframe(
                        pd.DataFrame(item["participant_results"]),
                        width="stretch",
                        hide_index=True,
                    )

    collecting_rounds = [
        item for item in rounds if item["status"] == "collecting"
    ]
    if collecting_rounds:
        with st.expander("Cancel an incomplete federated round"):
            selected_round = st.selectbox(
                "Round to cancel",
                collecting_rounds,
                format_func=lambda item: f"{item['round_id']} — {item['participants']} / 3 clients",
                key="cancel_federated_round_selection",
            )
            cancel_confirmed = st.checkbox(
                f"Cancel round {selected_round['round_id']}; submitted updates will not form a candidate.",
                key=f"cancel_round_confirmation_{selected_round['round_id']}",
            )
            if st.button(
                "Cancel selected round",
                disabled=not cancel_confirmed,
                key="cancel_selected_federated_round",
            ):
                try:
                    cancel_federated_round(principal, selected_round["round_id"])
                    st.success("Federated round cancelled; no candidate was activated.")
                    st.rerun()
                except (PermissionError, ValueError, OSError) as error:
                    st.error(f"Unable to cancel federated round: {error}")

    pending = [
        item["candidate"]
        for item in rounds
        if item.get("candidate") and item["candidate"].get("status") == "pending_approval"
    ]
    if pending:
        st.subheader("Inspect and approve a candidate")
        candidate = st.selectbox(
            "Candidate model",
            pending,
            format_func=lambda value: f"{value['candidate_id']} — {value['mode']}",
        )
        with st.container(border=True):
            st.markdown(f"#### {candidate['candidate_id']}")
            st.caption(f"{candidate['mode']} · pending explicit Admin review")
            with st.expander("Candidate details"):
                st.json(candidate)
        metrics = candidate["metrics"]
        metric_columns = st.columns(4)
        metric_columns[0].metric("Holdout accuracy", f"{metrics['accuracy']:.3f}")
        metric_columns[1].metric("Malignant recall", f"{metrics['malignant_recall']:.3f}")
        metric_columns[2].metric("F1 score", f"{metrics['f1_score']:.3f}")
        metric_columns[3].metric("ROC AUC", f"{metrics['roc_auc']:.3f}")
        st.caption(
            f"Evaluation uses {metrics['holdout_records']} held-out records. "
            "Scaling was fit only on training records."
        )
        if candidate.get("privacy_accounting"):
            accounting = candidate["privacy_accounting"]
            st.info(
                f"Conditional noisy-update accounting: epsilon ≤ "
                f"{accounting['epsilon_upper_bound']:.4f}, "
                f"delta={accounting['delta']:g}, across "
                f"{accounting['accounted_client_epoch_mechanisms']} submitted "
                "client/epoch mechanisms."
            )
            st.caption(accounting["accounting_scope"])
        approval = st.checkbox(
            f"Explicitly approve {candidate['candidate_id']} for hospital predictions.",
            key=f"round_approval_{candidate['candidate_id']}",
        )
        rejection = st.checkbox(
            f"Reject {candidate['candidate_id']} instead of deploying it.",
            key=f"round_rejection_{candidate['candidate_id']}",
        )
        approve_column, reject_column = st.columns(2)
        with approve_column:
            if st.button(
                "Approve and deploy",
                type="primary",
                disabled=not approval,
                key=f"deploy_{candidate['candidate_id']}",
                width="stretch",
            ):
                try:
                    deployed = review_candidate(
                        principal,
                        candidate["candidate_id"],
                        approve=True,
                    )
                    st.success(
                        f"Deployed {deployed['version']} for hospital predictions."
                    )
                    st.rerun()
                except (PermissionError, ValueError, FileNotFoundError, OSError) as error:
                    st.error(f"Deployment failed: {error}")
        with reject_column:
            if st.button(
                "Reject candidate",
                disabled=not rejection,
                key=f"reject_{candidate['candidate_id']}",
                width="stretch",
            ):
                try:
                    review_candidate(
                        principal,
                        candidate["candidate_id"],
                        approve=False,
                    )
                    st.success("Candidate rejected; the approved model was not changed.")
                    st.rerun()
                except (PermissionError, ValueError, OSError) as error:
                    st.error(f"Candidate rejection failed: {error}")

    st.subheader("Approved model")
    if active_model:
        st.success(
            f"{active_model.get('name', 'Approved model')} · "
            f"{active_model.get('version', 'current')}"
        )
        with st.expander("Approved model manifest"):
            st.json(active_model)
    else:
        st.warning("No approved model is currently deployed.")

    st.subheader("Privacy settings and limits")
    privacy = privacy_description(principal)
    st.write(f"**Experimental mode:** {privacy['dp_mode']}")
    st.write(f"**Server can observe:** {privacy['server_observes']}")
    st.warning(privacy["limitations"])

    st.subheader("Administrative audit log")
    log = audit_log(principal)
    if log.empty:
        st.info("No administrative events have been recorded.")
    else:
        with st.expander(f"View audit events ({len(log)})", expanded=True):
            st.dataframe(
                log,
                width="stretch",
                hide_index=True,
                column_config={
                    "details_json": st.column_config.TextColumn(
                        "Details",
                        width="medium",
                    ),
                },
            )


# ============================================================
# TAB 8: EXPERIMENTAL EVALUATION
# ============================================================

if active_page == "Experimental Evaluation":
    st.header("Experimental Evaluation")
    st.warning(
        "All clients are simulated in one process. Evaluation results are research "
        "measurements on WDBC, not clinical evidence."
    )

    st.subheader("IID versus Non-IID and privacy-noise utility")
    st.caption(
        "Runs sample-count-weighted FedAvg and the dashboard's noisy-client update "
        "path on the same training-only validation split. The final 114-record test "
        "partition is excluded from training, scaling, and this experiment."
    )
    if st.button(
        "Run IID / Non-IID and noise-utility evaluation",
        type="primary",
        key="run_iid_noniid_evaluation",
    ):
        try:
            with st.spinner("Training IID, Non-IID, and noisy client models..."):
                iid_results, iid_metadata = run_iid_noniid_validation()
            st.session_state.iid_noniid_live_results = iid_results
            st.session_state.iid_noniid_live_metadata = iid_metadata
        except (ValueError, OSError, RuntimeError) as error:
            st.error(f"IID / Non-IID evaluation failed: {error}")

    iid_results = st.session_state.get("iid_noniid_live_results")
    if iid_results is not None:
        st.caption("Newly measured in this dashboard session.")
        st.dataframe(
            format_metric_percentages_for_display(iid_results),
            width="stretch",
            hide_index=True,
        )
        iid_metadata = st.session_state.iid_noniid_live_metadata
        st.write(
            f"{iid_metadata['training_records']} fit records; "
            f"{iid_metadata['validation_records']} validation records; "
            f"scaler fit on fit records only; "
            f"{iid_metadata['rounds']} rounds × {iid_metadata['local_epochs']} local epochs."
        )
        st.write("Hospital partition sizes:", iid_metadata["hospital_partitions"])
        st.caption(iid_metadata["evaluation"])

    st.subheader("Cross-validation")
    st.caption(
        "The on-demand run evaluates cross-validation folds using the 455-record "
        "development partition only. Scaling is fit inside each fold; the final "
        "114-record test partition is not evaluated."
    )
    if st.button(
        "Run cross-validation now",
        type="primary",
        key="run_live_cross_validation",
    ):
        try:
            with tempfile.TemporaryDirectory() as temporary_directory:
                output_dir = Path(temporary_directory)
                _, central_summary = run_cross_validation(output_dir / "centralized")
                _, federated_summary = run_federated_cross_validation(
                    output_dir / "federated"
                )
            st.session_state.live_cv_summaries = {
                "centralized": central_summary,
                "federated": federated_summary,
            }
        except (ValueError, OSError, RuntimeError) as error:
            st.error(f"Cross-validation failed: {error}")

    live_cv = st.session_state.get("live_cv_summaries")
    if live_cv is not None:
        st.caption(
            "Newly measured in this dashboard session. Federated DP cross-validation "
            "uses the separate research gradient-aggregation path, not dashboard "
            "client-side DP-FedAvg."
        )
        st.write("Centralized Logistic Regression")
        st.dataframe(
            format_metric_percentages_for_display(live_cv["centralized"]),
            width="stretch",
            hide_index=True,
        )
        st.write("FedAvg and separate research DP gradient experiment")
        st.dataframe(
            format_metric_percentages_for_display(live_cv["federated"]),
            width="stretch",
            hide_index=True,
        )
    verified_dir = BASE / "research_outputs" / "verified"
    historical_files = (
        (
            "cross_validation_summary.csv",
            "Historical centralized cross-validation",
        ),
        (
            "federated_cross_validation_summary.csv",
            "Historical FedAvg / research DP cross-validation",
        ),
    )
    with st.expander("Previously stored cross-validation results"):
        historical_available = False
        for filename, title in historical_files:
            result_path = verified_dir / filename
            if result_path.is_file():
                historical_available = True
                st.markdown(f"**{title}**")
                st.dataframe(
                    pd.read_csv(result_path),
                    width="stretch",
                    hide_index=True,
                )
                st.caption(
                    "Previously stored result; it was not run by opening this dashboard."
                )
        if not historical_available:
            st.info("No historical cross-validation summary files are available.")

    st.subheader("Separate trust-weighting experiment")
    st.write(
        "Ordinary FedAvg weights client parameters by nᵢ / Σnⱼ. The separate "
        "trust-weighted research variant uses αᵢ = nᵢ × trustᵢ / "
        "Σ(nⱼ × trustⱼ), where trust is local validation balanced accuracy. "
        "It is not the live federated-round aggregation mode and is not a privacy "
        "mechanism."
    )
    trust_path = BASE / "research_outputs" / "verified" / "trust_aggregation_test_metrics.csv"
    trust_weights_path = BASE / "research_outputs" / "verified" / "trust_weights.csv"
    if trust_path.is_file() and trust_weights_path.is_file():
        st.caption("Previously stored trust experiment; it was not run by opening this dashboard.")
        st.dataframe(
            pd.read_csv(trust_weights_path),
            width="stretch",
            hide_index=True,
        )
        st.dataframe(
            pd.read_csv(trust_path),
            width="stretch",
            hide_index=True,
        )
    else:
        st.info("No stored trust-weighting results are available.")


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Federated Breast Cancer Research | "
    "Academic Research Prototype"
)
