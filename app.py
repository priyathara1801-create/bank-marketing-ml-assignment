import os
import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

st.set_page_config(
    page_title="Bank Marketing Classifier",
    page_icon="📊",
    layout="wide"
)

MODEL_DIR = "model"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "kNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest": "random_forest.pkl"
}

@st.cache_resource
def load_model(filename):
    return joblib.load(
        os.path.join(MODEL_DIR, filename)
    )

st.title("📊 Bank Marketing Classification")
st.write(
    "Interactive comparison of five classification models "
    "using the Bank Marketing dataset."
)

st.sidebar.header("Model Selection")

selected_model = st.sidebar.selectbox(
    "Choose a classification model",
    list(MODEL_FILES.keys())
)

uploaded_file = st.file_uploader(
    "Upload test data (CSV)",
    type=["csv"],
    help="Upload the test_data.csv file from this project."
)

if uploaded_file is None:
    st.info("Please upload test_data.csv to evaluate a model.")
    st.stop()

data = pd.read_csv(uploaded_file)

if "y" not in data.columns:
    st.error(
        "Invalid file. The CSV must contain the target column 'y'."
    )
    st.stop()

X = data.drop(columns=["y"])

y_true = (
    data["y"]
    .astype(str)
    .str.lower()
    .map({"yes": 1, "no": 0})
)

if y_true.isna().any():
    st.error(
        "The target column 'y' must contain only 'yes' or 'no'."
    )
    st.stop()

model = load_model(
    MODEL_FILES[selected_model]
)

pred = model.predict(X)
prob = model.predict_proba(X)[:, 1]

metrics = {
    "Accuracy": accuracy_score(y_true, pred),
    "AUC": roc_auc_score(y_true, prob),
    "Precision": precision_score(
        y_true, pred, zero_division=0
    ),
    "Recall": recall_score(
        y_true, pred, zero_division=0
    ),
    "F1 Score": f1_score(
        y_true, pred, zero_division=0
    ),
    "MCC": matthews_corrcoef(y_true, pred)
}

st.subheader(
    f"Evaluation Metrics — {selected_model}"
)

cols = st.columns(6)

for col, (name, value) in zip(
    cols,
    metrics.items()
):
    col.metric(
        name,
        f"{value:.4f}"
    )

left, right = st.columns(2)

with left:
    st.subheader("Confusion Matrix")

    cm = confusion_matrix(
        y_true,
        pred
    )

    cm_df = pd.DataFrame(
        cm,
        index=["Actual No", "Actual Yes"],
        columns=["Predicted No", "Predicted Yes"]
    )

    st.dataframe(
        cm_df,
        use_container_width=True
    )

with right:
    st.subheader("Classification Report")

    report = classification_report(
        y_true,
        pred,
        target_names=["No", "Yes"],
        output_dict=True,
        zero_division=0
    )

    st.dataframe(
        pd.DataFrame(report).transpose(),
        use_container_width=True
    )

st.subheader("Prediction Preview")

preview = X.copy()

preview["Actual y"] = data["y"].values

preview["Predicted y"] = [
    "yes" if value == 1 else "no"
    for value in pred
]

preview["Probability of yes"] = prob

st.dataframe(
    preview.head(25),
    use_container_width=True
)
