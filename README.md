# Explainable AI for Heart Disease Risk Prediction

BSc precursor project for the proposed MSc research direction *"Human-Centred Design
Frameworks for Explainable AI in Healthcare Applications."*

Full project design (background, research questions, methodology, evaluation plan,
roadmap) is in **[`docs/project_proposal.md`](docs/project_proposal.md)** — read that
first.

## What's in this repository

```
heart-xai-project/
├── README.md                          ← you are here
├── requirements.txt
├── data/                               ← put the dataset CSV here (see below)
├── notebooks/
│   └── heart_disease_xai_pipeline.ipynb   ← data cleaning, EDA, modelling, SHAP
├── app/
│   └── app.py                          ← Streamlit prototype + evaluation logging
└── docs/
    ├── project_proposal.md             ← full project design (all sections)
    ├── report_template.md              ← fill-in technical report structure
    └── evaluation_questionnaire.md     ← human-centred evaluation protocol
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Get the dataset

**Option A (recommended):** the notebook will fetch it automatically via `ucimlrepo`
(already in `requirements.txt`) — no manual download needed.

**Option B (manual):** download the Heart Disease dataset from the UCI Machine Learning
Repository (<https://archive.ics.uci.edu/dataset/45/heart+disease>) or a Kaggle mirror
(search "Heart Disease UCI"), and save it as `data/heart.csv` with columns:
`age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, target`

## Run the pipeline

```bash
jupyter notebook notebooks/heart_disease_xai_pipeline.ipynb
```

Run all cells top to bottom. This cleans the data, runs EDA, trains and compares four
models, generates SHAP explanations, and saves `app/artifacts/model_bundle.pkl` for the
prototype below.

## Run the prototype

```bash
cd app
streamlit run app.py
```

Opens in your browser. Enter patient-style data, click **Predict risk**, and view the
visual and/or plain-language explanation (toggle in the sidebar).

## Run the human evaluation

Follow the protocol in `docs/evaluation_questionnaire.md`. Participant responses are
logged automatically to `app/artifacts/evaluation_log.csv` as you use the app's built-in
"Research evaluation" section.

## Write up

Use `docs/report_template.md` as your report skeleton, filling in your own results only.

## A note on evidence

This project is designed so that every claim you make to a supervisor is backed by
something in this repository: real code, a real notebook run with real outputs, a real
prototype, and real (if small-scale) evaluation data. Commit to GitHub as you go rather
than all at once at the end — an incremental commit history is itself part of your
evidence of independent work. See the evidence-portfolio checklist in
`docs/project_proposal.md` (Section 18) for the full list of what to save.
