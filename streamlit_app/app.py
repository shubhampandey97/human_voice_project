import streamlit as st
import pandas as pd
import joblib
import numpy as np
from pathlib import Path

# ------------------ PATHS ------------------
BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"
SELECTOR_PATH = BASE_DIR / "models" / "selector.pkl"
FEATURE_PATH = BASE_DIR / "models" / "feature_names.pkl"

REPORTS_PATH = BASE_DIR / "reports"
FIGURES_PATH = REPORTS_PATH / "figures"

# ------------------ LOAD ------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    selector = joblib.load(SELECTOR_PATH)
    feature_names = joblib.load(FEATURE_PATH)
    return model, scaler, selector, feature_names

model, scaler, selector, feature_names = load_artifacts()

# ------------------ UI ------------------
st.set_page_config(page_title="Voice ML System", layout="wide")
st.title("🎙️ Human Voice Classification & Clustering")

st.sidebar.title("Navigation")
mode = st.sidebar.radio("Select Mode", ["Manual Input", "Upload CSV", "Clustering"])

# =========================
# 🔹 MANUAL INPUT
# =========================
if mode == "Manual Input":

    st.header("🎛️ Manual Feature Input")

    inputs = {}

    # Dynamic input generation
    cols = st.columns(3)

    for i, feature in enumerate(feature_names):
        with cols[i % 3]:
            inputs[feature] = st.number_input(
                feature,
                value=0.0,
                format="%.4f"
            )

    if st.button("🚀 Predict"):

        try:
            input_df = pd.DataFrame([inputs])

            # Pipeline
            X_scaled = scaler.transform(input_df)
            X_selected = selector.transform(X_scaled)

            prediction = model.predict(X_selected)[0]

            result = "Male" if prediction == 1 else "Female"

            st.success(f"Prediction: {result}")

        except Exception as e:
            st.error(f"Error: {e}")

# =========================
# 🔹 CSV MODE
# =========================
elif mode == "Upload CSV":

    st.header("📂 Batch Prediction")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        st.write("Preview", df.head())

        try:
            X_scaled = scaler.transform(df)
            X_selected = selector.transform(X_scaled)

            predictions = model.predict(X_selected)

            df["Prediction"] = ["Male" if p == 1 else "Female" for p in predictions]

            st.dataframe(df)

            st.bar_chart(df["Prediction"].value_counts())

        except Exception as e:
            st.error(f"Error: {e}")

# =========================
# 🔹 CLUSTERING
# =========================
elif mode == "Clustering":

    st.header("📊 Clustering Insights")

    st.metric("Best K", "2")
    st.metric("Silhouette Score", "0.295")

    elbow_path = FIGURES_PATH / "elbow_method.png"
    if elbow_path.exists():
        st.image(str(elbow_path))

    kmeans_path = FIGURES_PATH / "KMeans_clusters.png"
    if kmeans_path.exists():
        st.image(str(kmeans_path))