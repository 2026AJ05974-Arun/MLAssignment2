"""Streamlit app to evaluate the saved letter recognition models on a test CSV."""

import html
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

from train_models import evaluate_model

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
TEST_DATA = BASE_DIR / "test_data.csv"
TARGET = "Class"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes (Gaussian)": "gaussian_naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}


@st.cache_resource
def load_model(filename):
    return joblib.load(MODEL_DIR / filename)


@st.cache_data(show_spinner=False)
def evaluate_all(data):
    X, y = data.drop(columns=TARGET), data[TARGET]
    with ThreadPoolExecutor() as pool:
        futures = {
            name: pool.submit(evaluate_model, load_model(f), X, y)
            for name, f in MODEL_FILES.items()
        }
        return {name: fut.result() for name, fut in futures.items()}


@st.cache_data(show_spinner=False)
def predict(name, data):
    return load_model(MODEL_FILES[name]).predict(data.drop(columns=TARGET))


def heatmap_styles(cm):
    """Green shades on the diagonal (correct), red shades off it (mistakes)."""
    values = cm.to_numpy()
    diag_max = max(values.diagonal().max(), 1)
    off = values.copy()
    for i in range(len(off)):
        off[i, i] = 0
    off_max = max(off.max(), 1)
    styles = pd.DataFrame("", index=cm.index, columns=cm.columns)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values[i, j]
            if v == 0:
                continue
            if i == j:
                alpha = 0.15 + 0.85 * v / diag_max
                styles.iat[i, j] = f"background-color: rgba(52,168,83,{alpha:.2f}); color: #FFFFFF; font-weight: 600"
            else:
                alpha = 0.1 + 0.8 * v / off_max
                styles.iat[i, j] = f"background-color: rgba(234,67,53,{alpha:.2f})"
    return styles


GOOGLE_COLORS = ["#4285F4", "#EA4335", "#FBBC05", "#4285F4", "#34A853", "#EA4335"]

GOOGLE_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Google+Sans:wght@400;500;700&family=Roboto:wght@400;500&display=swap');

html, body, [class*="css"], .stMarkdown, .stText, button, input, select, textarea {
    font-family: 'Google Sans', 'Roboto', Arial, sans-serif !important;
}
.block-container { padding-top: 2rem; max-width: 1400px; }
h1, h2, h3 { font-family: 'Google Sans', 'Roboto', sans-serif !important; color: #202124; font-weight: 500 !important; }

.g-header { display: flex; align-items: baseline; gap: 0.75rem; margin-bottom: 0.25rem; }
.g-logo { font-size: 2.6rem; font-weight: 500; letter-spacing: -1px; }
.g-sub { font-size: 1.6rem; color: #5F6368; font-weight: 400; }
.g-desc { color: #5F6368; font-size: 1rem; margin-bottom: 1.5rem; }

.g-chip {
    display: inline-block; background: #E8F0FE; color: #1967D2; border-radius: 16px;
    padding: 4px 14px; font-size: 0.875rem; font-weight: 500; margin-right: 8px;
}

.g-card {
    background: #FFFFFF; border: 1px solid #DADCE0; border-radius: 12px;
    padding: 16px 18px; box-shadow: 0 1px 2px rgba(60,64,67,.3), 0 1px 3px 1px rgba(60,64,67,.15);
    border-top: 4px solid var(--accent); height: 100%;
}
.g-card-label { color: #5F6368; font-size: 0.8rem; text-transform: uppercase; letter-spacing: .6px; }
.g-card-value { color: #202124; font-size: 1.7rem; font-weight: 500; margin-top: 4px; }
.g-card-delta { font-size: 0.8rem; margin-top: 2px; }
.g-best { color: #137333; } .g-behind { color: #C5221F; }

.stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid #DADCE0; }
.stTabs [data-baseweb="tab"] { padding: 10px 18px; font-weight: 500; color: #5F6368; }
.stTabs [aria-selected="true"] { color: #1A73E8 !important; }

.g-section {
    font-size: 1.25rem; font-weight: 500; color: #202124; margin: 1.75rem 0 0.75rem 0;
    padding-bottom: 6px; border-bottom: 1px solid #E8EAED;
}

section[data-testid="stSidebar"] { background: #F8F9FA; border-right: 1px solid #DADCE0; }
.stButton > button, .stDownloadButton > button {
    background: #1A73E8; color: #FFFFFF; border: none; border-radius: 20px;
    padding: 0.45rem 1.4rem; font-weight: 500;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    background: #1765CC; color: #FFFFFF; box-shadow: 0 1px 3px rgba(60,64,67,.3);
}
div[data-testid="stDataFrame"] { border: 1px solid #DADCE0; border-radius: 8px; }
</style>
"""


def section(title):
    st.markdown(f'<div class="g-section">{title}</div>', unsafe_allow_html=True)


st.set_page_config(page_title="Letter Recognition", page_icon="🔤", layout="wide")
st.markdown(GOOGLE_CSS, unsafe_allow_html=True)

logo = "".join(
    f'<span style="color:{GOOGLE_COLORS[i % len(GOOGLE_COLORS)]}">{ch}</span>'
    for i, ch in enumerate("Letters")
)
st.markdown(
    f'<div class="g-header"><span class="g-logo">{logo}</span>'
    f'<span class="g-sub">Model Comparison</span></div>'
    '<div class="g-desc">Pick a model, upload a test CSV (or use the bundled one) and see how '
    "the saved models score. Nothing is retrained here.</div>",
    unsafe_allow_html=True,
)

default_data = pd.read_csv(TEST_DATA)
feature_cols = [c for c in default_data.columns if c != TARGET]

with st.sidebar:
    st.markdown("### Settings")
    uploaded = st.file_uploader("Upload test CSV", type="csv")
    st.caption(f"Needs the {len(feature_cols)} feature columns and a '{TARGET}' column.")
    selected = st.selectbox("Model", list(MODEL_FILES))
    st.download_button(
        "Download sample CSV", default_data.to_csv(index=False), "test_data.csv"
    )

if uploaded is None:
    data = default_data
    source = "bundled test_data.csv"
else:
    try:
        data = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read that file: {e}")
        st.stop()
    source = uploaded.name

missing = [c for c in feature_cols + [TARGET] if c not in data.columns]
if missing:
    st.error("Missing columns: " + ", ".join(missing))
    st.stop()
if data[feature_cols + [TARGET]].isna().any().any():
    st.error("The CSV has missing values.")
    st.stop()
if not all(pd.api.types.is_numeric_dtype(data[c]) for c in feature_cols):
    st.error("All feature columns must be numeric.")
    st.stop()

data = data[feature_cols + [TARGET]]
y = data[TARGET]

letters = list(load_model(MODEL_FILES[selected]).classes_)
if not set(y.unique()) <= set(letters):
    st.error("The Class column has labels that are not letters A-Z.")
    st.stop()
if y.nunique() < 26:
    st.warning("This file does not contain all 26 letters, so AUC is averaged over the letters present.")

with st.spinner("Evaluating models..."):
    results = evaluate_all(data)
    pred = predict(selected, data)

table = pd.DataFrame(results).T
table.index.name = "ML Model Name"
best_model = table["F1"].idxmax()

st.markdown(
    f'<span class="g-chip">Data: {html.escape(source)}</span>'
    f'<span class="g-chip">{len(data):,} rows</span>'
    f'<span class="g-chip">{y.nunique()} letters</span>'
    f'<span class="g-chip">Best F1: {html.escape(best_model)}</span>',
    unsafe_allow_html=True,
)

section(selected)
accents = ["#4285F4", "#EA4335", "#FBBC05", "#34A853", "#4285F4", "#EA4335"]
cols = st.columns(6)
for col, accent, (metric, value) in zip(cols, accents, results[selected].items()):
    gap = value - table[metric].max()
    if gap >= 0:
        delta = '<div class="g-card-delta g-best">Best of 5 models</div>'
    else:
        delta = f'<div class="g-card-delta g-behind">{gap:+.4f} vs best</div>'
    col.markdown(
        f'<div class="g-card" style="--accent:{accent}">'
        f'<div class="g-card-label">{html.escape(str(metric))}</div>'
        f'<div class="g-card-value">{value:.4f}</div>{delta}</div>',
        unsafe_allow_html=True,
    )

st.write("")
tab_compare, tab_cm, tab_letters, tab_data = st.tabs(
    ["Model comparison", "Confusion matrix", "Per-letter report", "Data preview"]
)

with tab_compare:
    st.dataframe(
        table.style.format("{:.4f}").highlight_max(
            axis=0, props="background-color:#E6F4EA;color:#137333;font-weight:600"
        ),
        width="stretch",
    )
    metric = st.segmented_control(
        "Metric to chart", list(table.columns), default="Accuracy", key="chart_metric"
    ) or "Accuracy"
    chart = table[[metric]].sort_values(metric, ascending=False)
    st.bar_chart(chart, horizontal=True, color="#1A73E8", height=280)

with tab_cm:
    cm = pd.DataFrame(
        confusion_matrix(y, pred, labels=letters),
        index=[f"Actual {c}" for c in letters],
        columns=[f"Pred {c}" for c in letters],
    )
    left, right = st.columns([3, 1])
    with left:
        st.caption("Green = correct predictions, red = mistakes. Darker means more rows.")
        st.dataframe(cm.style.apply(heatmap_styles, axis=None), width="stretch", height=600)
    with right:
        st.markdown("**Most common mistakes**")
        mistakes = (
            pd.DataFrame({"Actual": y.to_numpy(), "Predicted": pred})
            .query("Actual != Predicted")
            .value_counts()
            .head(10)
            .rename("Count")
            .reset_index()
        )
        st.dataframe(mistakes, hide_index=True, width="stretch")

with tab_letters:
    report = pd.DataFrame(
        classification_report(y, pred, labels=letters, zero_division=0, output_dict=True)
    ).T
    per_letter = report.loc[letters]
    st.markdown("**F1 score per letter**")
    st.bar_chart(per_letter["f1-score"], color="#34A853", height=260)
    st.dataframe(
        report.round(4).style.format(precision=4).highlight_min(
            subset=["f1-score"], props="background-color:#FCE8E6;color:#C5221F;font-weight:600"
        ),
        width="stretch",
        height=500,
    )

with tab_data:
    st.dataframe(data.head(100), width="stretch", height=400)
    st.caption(f"Showing the first {min(100, len(data))} of {len(data):,} rows.")
