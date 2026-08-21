"""
Streamlit prototype: Explainable AI for Heart Disease Risk Prediction
----------------------------------------------------------------------
Run with:  streamlit run app.py   (from inside the app/ folder)

Requires artifacts/model_bundle.pkl, produced by running
notebooks/heart_disease_xai_pipeline.ipynb first.

This app is a RESEARCH PROTOTYPE for a BSc project. It is NOT a medical
device and must not be used to inform real clinical decisions.
"""

import csv
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import streamlit as st

# ----------------------------------------------------------------------
# Config & artifact loading
# ----------------------------------------------------------------------
APP_DIR = Path(__file__).parent
ARTIFACT_PATH = APP_DIR / "artifacts" / "model_bundle.pkl"
LOG_PATH = APP_DIR / "artifacts" / "evaluation_log.csv"

st.set_page_config(page_title="Heart Disease Risk & Explanation Prototype", layout="wide")

CATEGORICAL_LABELS = {
    "sex": {"Female": 0, "Male": 1},
    "cp": {
        "Typical angina": 0,
        "Atypical angina": 1,
        "Non-anginal pain": 2,
        "Asymptomatic": 3,
    },
    "fbs": {"No (<= 120 mg/dl)": 0, "Yes (> 120 mg/dl)": 1},
    "restecg": {
        "Normal": 0,
        "ST-T wave abnormality": 1,
        "Left ventricular hypertrophy": 2,
    },
    "exang": {"No": 0, "Yes": 1},
    "slope": {"Upsloping": 0, "Flat": 1, "Downsloping": 2},
    "thal": {"Normal": 3, "Fixed defect": 6, "Reversible defect": 7},
}

FEATURE_HELP = {
    "age": "Age in years",
    "sex": "Biological sex as recorded in the dataset",
    "cp": "Type of chest pain reported",
    "trestbps": "Resting blood pressure (mm Hg) on hospital admission",
    "chol": "Serum cholesterol (mg/dl)",
    "fbs": "Whether fasting blood sugar exceeds 120 mg/dl",
    "restecg": "Resting electrocardiographic result",
    "thalach": "Maximum heart rate achieved during exercise test",
    "exang": "Whether exercise induced angina (chest pain)",
    "oldpeak": "ST depression induced by exercise, relative to rest",
    "slope": "Slope of the peak exercise ST segment",
    "ca": "Number of major vessels (0-3) coloured by fluoroscopy",
    "thal": "Thalassemia test result",
}


@st.cache_resource
def load_bundle():
    if not ARTIFACT_PATH.exists():
        return None
    with open(ARTIFACT_PATH, "rb") as f:
        return pickle.load(f)


def get_positive_class_shap(shap_values):
    arr = np.array(shap_values)
    if isinstance(shap_values, list):
        return shap_values[1]
    if arr.ndim == 3:
        return arr[:, :, 1]
    return arr


@st.cache_resource
def build_explainer(_bundle):
    model = _bundle["model"]
    if _bundle["model_name"] in ("RandomForest", "GradientBoosting"):
        return shap.TreeExplainer(model), "tree"
    background = _bundle["background_sample"]
    return shap.KernelExplainer(model.predict_proba, background), "kernel"


def compute_explanation(bundle, explainer, explainer_kind, X_row_transformed):
    if explainer_kind == "tree":
        raw = explainer.shap_values(X_row_transformed)
        base_value = explainer.expected_value
        base_value = (
            base_value[1]
            if isinstance(base_value, (list, np.ndarray)) and np.ndim(base_value) > 0
            else base_value
        )
    else:
        raw = explainer.shap_values(X_row_transformed, nsamples=200)
        base_value = explainer.expected_value
        base_value = base_value[1] if np.ndim(base_value) > 0 else base_value

    shap_vals = get_positive_class_shap(raw)[0]
    return shap_vals, base_value


def text_explanation(feature_cols, shap_vals, raw_row, top_n=3):
    order = np.argsort(-np.abs(shap_vals))[:top_n]
    lines = []
    for i in order:
        feat = feature_cols[i]
        val = raw_row[feat].values[0]
        direction = "increased" if shap_vals[i] > 0 else "decreased"
        lines.append(f"- **{feat}** = {val} {direction} the predicted risk")
    return lines


# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------
st.sidebar.title("About this prototype")
st.sidebar.info(
    "Research prototype for a BSc project on explainable AI in healthcare. "
    "Uses a model trained on the UCI Heart Disease dataset. "
    "**Not a medical device — for research/demonstration only.**"
)

explanation_format = st.sidebar.radio(
    "Explanation format to display",
    ["Visual (SHAP chart)", "Plain-language text", "Both"],
    index=2,
    help="Used to compare explanation formats in the human-centred evaluation study.",
)

st.sidebar.divider()
st.sidebar.caption(
    "If you are a study participant, please leave this set to whatever the "
    "researcher asked you to use for this task."
)

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
st.title("Heart Disease Risk Prediction — Explainable AI Prototype")

bundle = load_bundle()
if bundle is None:
    st.error(
        "No trained model found. Run `notebooks/heart_disease_xai_pipeline.ipynb` "
        "first — it saves `app/artifacts/model_bundle.pkl`, which this app needs."
    )
    st.stop()

feature_cols = bundle["feature_cols"]

st.subheader("1. Enter patient information")
col1, col2, col3 = st.columns(3)
inputs = {}
with col1:
    inputs["age"] = st.number_input("Age", 18, 100, 54, help=FEATURE_HELP["age"])
    inputs["sex"] = CATEGORICAL_LABELS["sex"][
        st.selectbox("Sex", list(CATEGORICAL_LABELS["sex"].keys()), help=FEATURE_HELP["sex"])
    ]
    inputs["cp"] = CATEGORICAL_LABELS["cp"][
        st.selectbox("Chest pain type", list(CATEGORICAL_LABELS["cp"].keys()), help=FEATURE_HELP["cp"])
    ]
    inputs["trestbps"] = st.number_input("Resting blood pressure (mm Hg)", 80, 220, 130, help=FEATURE_HELP["trestbps"])
    inputs["chol"] = st.number_input("Serum cholesterol (mg/dl)", 100, 600, 246, help=FEATURE_HELP["chol"])
with col2:
    inputs["fbs"] = CATEGORICAL_LABELS["fbs"][
        st.selectbox("Fasting blood sugar", list(CATEGORICAL_LABELS["fbs"].keys()), help=FEATURE_HELP["fbs"])
    ]
    inputs["restecg"] = CATEGORICAL_LABELS["restecg"][
        st.selectbox("Resting ECG", list(CATEGORICAL_LABELS["restecg"].keys()), help=FEATURE_HELP["restecg"])
    ]
    inputs["thalach"] = st.number_input("Max heart rate achieved", 60, 220, 150, help=FEATURE_HELP["thalach"])
    inputs["exang"] = CATEGORICAL_LABELS["exang"][
        st.selectbox("Exercise-induced angina", list(CATEGORICAL_LABELS["exang"].keys()), help=FEATURE_HELP["exang"])
    ]
    inputs["oldpeak"] = st.number_input("ST depression (oldpeak)", 0.0, 7.0, 1.0, step=0.1, help=FEATURE_HELP["oldpeak"])
with col3:
    inputs["slope"] = CATEGORICAL_LABELS["slope"][
        st.selectbox("Slope of peak exercise ST segment", list(CATEGORICAL_LABELS["slope"].keys()), help=FEATURE_HELP["slope"])
    ]
    inputs["ca"] = st.selectbox("Major vessels coloured by fluoroscopy (0-3)", [0, 1, 2, 3], help=FEATURE_HELP["ca"])
    inputs["thal"] = CATEGORICAL_LABELS["thal"][
        st.selectbox("Thalassemia result", list(CATEGORICAL_LABELS["thal"].keys()), help=FEATURE_HELP["thal"])
    ]

raw_row = pd.DataFrame([inputs])[feature_cols]

if st.button("Predict risk", type="primary"):
    model = bundle["model"]
    if bundle["needs_scaling"]:
        X_transformed = pd.DataFrame(
            bundle["scaler"].transform(raw_row), columns=feature_cols
        )
    else:
        X_transformed = raw_row

    proba = model.predict_proba(X_transformed)[0, 1]
    st.session_state["last_proba"] = proba
    st.session_state["last_raw_row"] = raw_row
    st.session_state["last_X_transformed"] = X_transformed

if "last_proba" in st.session_state:
    st.subheader("2. Prediction")
    proba = st.session_state["last_proba"]
    risk_label = "Higher risk" if proba >= 0.5 else "Lower risk"
    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric("Predicted probability of disease", f"{proba:.0%}")
        st.write(f"**Category:** {risk_label} (threshold 50%)")
    with c2:
        st.progress(min(max(proba, 0.0), 1.0))
        st.caption(
            f"Model used: {bundle['model_name']} · "
            f"Test-set ROC-AUC when trained: {bundle['test_metrics']['roc_auc']:.2f}"
        )

    st.subheader("3. Explanation")
    explainer, explainer_kind = build_explainer(bundle)
    shap_vals, base_value = compute_explanation(
        bundle, explainer, explainer_kind, st.session_state["last_X_transformed"]
    )

    if explanation_format in ("Visual (SHAP chart)", "Both"):
        st.markdown("**Visual explanation** — how each factor pushed the prediction "
                     "up (red) or down (blue) from the average patient.")
        explanation_obj = shap.Explanation(
            values=shap_vals,
            base_values=base_value,
            data=st.session_state["last_raw_row"].iloc[0].values,
            feature_names=feature_cols,
        )
        fig = plt.figure(figsize=(8, 4.5))
        shap.plots.waterfall(explanation_obj, show=False)
        st.pyplot(fig, clear_figure=True)

    if explanation_format in ("Plain-language text", "Both"):
        st.markdown("**Plain-language explanation** — the three factors that "
                     "influenced this prediction the most:")
        for line in text_explanation(feature_cols, shap_vals, st.session_state["last_raw_row"]):
            st.markdown(line)

    # ------------------------------------------------------------------
    # Human-centred evaluation logging (for the small user study)
    # ------------------------------------------------------------------
    with st.expander("Research evaluation — for study participants only"):
        st.write(
            "If you are taking part in the usability study, please answer the "
            "questions below about the explanation you just saw."
        )
        top_feature = feature_cols[int(np.argmax(np.abs(shap_vals)))]
        distractors = [f for f in feature_cols if f != top_feature]
        options = [top_feature] + list(np.random.default_rng(0).choice(distractors, 2, replace=False))
        options = list(pd.Series(options).sample(frac=1, random_state=None))

        participant_id = st.text_input("Participant ID (e.g. P1, P2...)")
        comprehension_answer = st.radio(
            "Which factor had the SINGLE largest effect on this prediction?", options
        )
        trust_rating = st.slider("I trust this prediction (1 = strongly disagree, 5 = strongly agree)", 1, 5, 3)
        clarity_rating = st.slider("This explanation was easy to understand (1 = strongly disagree, 5 = strongly agree)", 1, 5, 3)
        comments = st.text_area("Any other comments (optional)")

        if st.button("Submit evaluation response"):
            LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            file_exists = LOG_PATH.exists()
            with open(LOG_PATH, "a", newline="") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow([
                        "timestamp", "participant_id", "explanation_format",
                        "predicted_probability", "top_feature", "comprehension_answer",
                        "comprehension_correct", "trust_rating", "clarity_rating", "comments",
                    ])
                writer.writerow([
                    datetime.now().isoformat(timespec="seconds"),
                    participant_id,
                    explanation_format,
                    round(float(proba), 4),
                    top_feature,
                    comprehension_answer,
                    comprehension_answer == top_feature,
                    trust_rating,
                    clarity_rating,
                    comments,
                ])
            st.success(f"Response saved to {LOG_PATH.name}. Thank you!")
else:
    st.info("Fill in the patient information above and click **Predict risk** to continue.")
