import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve

st.set_page_config(page_title="Loan Approval Prediction", page_icon="🏦", layout="wide")

DATA_FILE = Path(__file__).parent / "loan_approval_dataset.csv"
FEATURES = [
    "no_of_dependents", "education", "self_employed", "income_annum", "loan_amount",
    "loan_term", "cibil_score", "residential_assets_value", "commercial_assets_value",
    "luxury_assets_value", "bank_asset_value"
]
MODEL_NAMES = ["Random Forest", "Logistic Regression", "SVC"]



@st.cache_data(show_spinner=False)
def load_data(file_bytes=None):
    if file_bytes is not None:
        df = pd.read_csv(file_bytes)
    else:
        if not DATA_FILE.exists():
            return None
        df = pd.read_csv(DATA_FILE)
    df.columns = df.columns.str.strip()
    df = df.drop(columns=["loan_id", "Loan_ID", "loan_id "], errors="ignore")
    if "loan_status" not in df.columns:
        raise ValueError("Dataset must contain a 'loan_status' target column.")
    df["loan_status"] = df["loan_status"].astype(str).str.strip().str.title()
    df = df[df["loan_status"].isin(["Approved", "Rejected"])].copy()
    for c in FEATURES:
        if c not in df.columns:
            raise ValueError(f"Missing required feature column: {c}")
    return df[FEATURES + ["loan_status"]].dropna()


@st.cache_resource(show_spinner="Training models...")
def train_models(df):
    X = df[FEATURES].copy()
    # Match the project's earlier preprocessing: label-encode categorical columns.
    X["education"] = X["education"].astype(str).str.strip().map({"Graduate": 0, "Not Graduate": 1})
    X["self_employed"] = X["self_employed"].astype(str).str.strip().map({"No": 0, "Yes": 1})
    y = (df["loan_status"] == "Approved").astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    model_defs = {
        "Random Forest": RandomForestClassifier(random_state=1),
        "Logistic Regression": LogisticRegression(max_iter=5000),
        "SVC": SVC(probability=True, random_state=1)
    }
    trained, metrics, predictions, probabilities = {}, {}, {}, {}
    for name, estimator in model_defs.items():
        estimator.fit(X_train, y_train)
        pred = estimator.predict(X_test)
        prob = estimator.predict_proba(X_test)[:, 1]
        trained[name] = estimator
        metrics[name] = {
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred, zero_division=0),
            "F1 Score": f1_score(y_test, pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, prob)
        }
        predictions[name] = pred
        probabilities[name] = prob
    return trained, metrics, X_test, y_test, predictions, probabilities




st.title("🏦 Loan Approval Prediction System")
st.caption("Interactive machine-learning demo: predict loan status and compare three classifiers.")
with st.sidebar:
    st.header("Project Controls")
    uploaded = st.file_uploader("Upload dataset (optional)", type=["csv"])
    st.caption("If no file is uploaded, the app reads loan_approval_dataset.csv from this folder.")
    page = st.radio("Navigate", ["Dashboard", "Loan Prediction", "Compare Models", "Model Analytics", "About"])
    threshold = st.slider("Approval decision threshold", 0.05, 0.95, 0.50, 0.01,
                        help="Predicted approval probability at or above this value is classified as Approved.")

try:
    df = load_data(uploaded)
except Exception as e:
    st.error(f"Dataset error: {e}")
    st.stop()
if df is None:
    st.error("Dataset not found. Place your CSV in this project folder and name it loan_approval_dataset.csv, or upload it in the sidebar.")
    st.stop()
models, metrics, X_test, y_test, preds, probs = train_models(df)
metrics_df = pd.DataFrame(metrics).T


if page == "Dashboard":
    st.subheader("Project Dashboard")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Applications", f"{len(df):,}")
    c2.metric("Approved", f"{(df.loan_status == 'Approved').sum():,}")
    c3.metric("Rejected", f"{(df.loan_status == 'Rejected').sum():,}")
    c4.metric("Features", len(FEATURES))
    st.markdown("### Dataset Outcome Distribution")
    counts = df.loan_status.value_counts().rename_axis("Loan Status").reset_index(name="Applications")
    fig = px.bar(counts, x="Loan Status", y="Applications", color="Loan Status", text="Applications")
    fig.update_layout(height=360, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("### Model Performance Snapshot")
    perf = metrics_df.sort_values("Accuracy", ascending=False)
    st.dataframe(perf.style.format("{:.3f}"), use_container_width=True)
    st.info("Models are trained on an 80/20 train-test split. Results may vary slightly if the dataset changes.")

elif page == "Loan Prediction":
    st.subheader("Applicant Information")
    left, right = st.columns(2)
    with left:
        dependents = st.number_input("Number of Dependents", min_value=0, max_value=20, value=2)
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        employed = st.selectbox("Self-Employed", ["Yes", "No"])
        income = st.number_input("Annual Income (₹)", min_value=0, value=5000000, step=100000)
        loan_amt = st.number_input("Loan Amount (₹)", min_value=0, value=10000000, step=100000)
        term = st.number_input("Loan Term (years)", min_value=1, max_value=50, value=10)
        cibil = st.slider("CIBIL Score", 300, 900, 750)
    with right:
        residential = st.number_input("Residential Assets Value (₹)", min_value=0, value=5000000, step=100000)
        commercial = st.number_input("Commercial Assets Value (₹)", min_value=0, value=2000000, step=100000)
        luxury = st.number_input("Luxury Assets Value (₹)", min_value=0, value=3000000, step=100000)
        bank = st.number_input("Bank Asset Value (₹)", min_value=0, value=2000000, step=100000)
        selected = st.selectbox("Model", MODEL_NAMES, index=0)
        st.markdown(" ")
        run = st.button("🔍 Predict Loan Status", type="primary", use_container_width=True)
    applicant = pd.DataFrame([{
        "no_of_dependents": dependents, "education": education, "self_employed": employed,
        "income_annum": income, "loan_amount": loan_amt, "loan_term": term,
        "cibil_score": cibil, "residential_assets_value": residential,
        "commercial_assets_value": commercial, "luxury_assets_value": luxury,
        "bank_asset_value": bank
    }], columns=FEATURES)
    if run:
        applicant["education"] = applicant["education"].map({"Graduate": 0, "Not Graduate": 1})
        applicant["self_employed"] = applicant["self_employed"].map({"No": 0, "Yes": 1})
        probability = float(models[selected].predict_proba(applicant)[0, 1])
        status = "Approved" if probability >= threshold else "Rejected"
        st.markdown("---")
        st.subheader("Prediction Result")
        color = "success" if status == "Approved" else "error"
        getattr(st, color)(f"### {status}")
        a, b, c = st.columns(3)
        a.metric("Selected Model", selected)
        b.metric("Approval Probability", f"{probability:.1%}")
        c.metric("Decision Threshold", f"{threshold:.0%}")
        st.progress(min(max(probability, 0.0), 1.0))
        st.write(f"Rejected probability: **{1-probability:.1%}**")
        st.caption("This is an educational model output, not a real lending decision. Probability estimates are not guarantees.")
        with st.expander("View submitted applicant profile"):
            st.dataframe(applicant, use_container_width=True)

elif page == "Compare Models":
    st.subheader("Compare Models on the Same Applicant")
    st.write("Enter applicant details, then view each classifier's output side-by-side.")
    col1, col2, col3 = st.columns(3)
    with col1:
        d = st.number_input("Dependents", 0, 20, 2, key="cmp_dep")
        edu = st.selectbox("Education", ["Graduate", "Not Graduate"], key="cmp_edu")
        emp = st.selectbox("Self-employed", ["Yes", "No"], key="cmp_emp")
        inc = st.number_input("Annual income (₹)", min_value=0, value=5000000, step=100000, key="cmp_inc")
    with col2:
        la = st.number_input("Loan amount (₹)", min_value=0, value=10000000, step=100000, key="cmp_la")
        lt = st.number_input("Loan term (years)", min_value=1, max_value=50, value=10, key="cmp_lt")
        cs = st.slider("CIBIL score", 300, 900, 750, key="cmp_cs")
        ra = st.number_input("Residential assets (₹)", min_value=0, value=5000000, step=100000, key="cmp_ra")
    with col3:
        ca = st.number_input("Commercial assets (₹)", min_value=0, value=2000000, step=100000, key="cmp_ca")
        lux = st.number_input("Luxury assets (₹)", min_value=0, value=3000000, step=100000, key="cmp_lux")
        ba = st.number_input("Bank assets (₹)", min_value=0, value=2000000, step=100000, key="cmp_ba")
    applicant2 = pd.DataFrame([{
        "no_of_dependents": d, "education": edu, "self_employed": emp,
        "income_annum": inc, "loan_amount": la, "loan_term": lt,
        "cibil_score": cs, "residential_assets_value": ra,
        "commercial_assets_value": ca, "luxury_assets_value": lux,
        "bank_asset_value": ba
    }], columns=FEATURES)
    if st.button("⚖️ Compare All Three Models", type="primary"):
        applicant2["education"] = applicant2["education"].map({"Graduate": 0, "Not Graduate": 1})
        applicant2["self_employed"] = applicant2["self_employed"].map({"No": 0, "Yes": 1})
        rows = []
        for name in MODEL_NAMES:
            p = float(models[name].predict_proba(applicant2)[0, 1])
            pred = "Approved" if p >= threshold else "Rejected"
            rows.append({"Model": name, "Prediction": pred, "Approval Probability": p, "Rejected Probability": 1-p})
        comparison = pd.DataFrame(rows)
        st.markdown("### Model Predictions")
        st.dataframe(comparison.style.format({"Approval Probability": " {:.1%}", "Rejected Probability": " {:.1%}"}), use_container_width=True)
        st.plotly_chart(px.bar(comparison, x="Model", y="Approval Probability", text_auto=".1%", range_y=[0,1]), use_container_width=True)
        approved_votes = sum(comparison.Prediction == "Approved")
        st.info(f"Model consensus: **{approved_votes}/3** models predict approval at the current threshold ({threshold:.0%}).")

elif page == "Model Analytics":
    st.subheader("Model Evaluation Analytics")
    selected = st.selectbox("Select model", MODEL_NAMES, key="analytics_model")
    m = metrics[selected]
    cols = st.columns(5)
    for col, metric in zip(cols, ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]):
        col.metric(metric, f"{m[metric]:.3f}")
    st.markdown("### Confusion Matrix")
    cm = confusion_matrix(y_test, preds[selected], labels=[0,1])
    cm_df = pd.DataFrame(cm, index=["Actual Rejected", "Actual Approved"], columns=["Predicted Rejected", "Predicted Approved"])
    fig_cm = px.imshow(cm_df, text_auto=True, aspect="auto", labels=dict(x="Predicted", y="Actual", color="Count"))
    st.plotly_chart(fig_cm, use_container_width=True)
    st.markdown("### ROC Curve Comparison")
    fig = go_roc = None
    fig, ax = plt.subplots()
    for name in MODEL_NAMES:
        fpr, tpr, _ = roc_curve(y_test, probs[name])
        ax.plot(fpr, tpr, label=f"{name} (AUC={metrics[name]['ROC-AUC']:.3f})")
    ax.plot([0,1], [0,1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend()
    st.pyplot(fig)
    st.markdown("### Metrics Comparison")
    st.dataframe(metrics_df.style.format("{:.3f}"), use_container_width=True)

elif page == "About":
    st.subheader("About This Project")
    st.markdown("**Objective:** predict loan application status (Approved or Rejected) using applicant, credit, loan and asset-related information.")
    st.markdown("**Models:** Random Forest, Logistic Regression and Support Vector Classifier (SVC).")
    st.markdown("**Dataset:** Kaggle loan approval dataset supplied for this project.")
    st.warning("This application is an academic demonstration. It should not be used to make real lending decisions. Loan decisions require fairness, compliance, explainability and human review.")
