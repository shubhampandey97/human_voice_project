import sys
from pathlib import Path

# ------------------ PATH FIX ------------------
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))

import streamlit as st
import pandas as pd
import joblib
import numpy as np

from src.features.build_features import build_features

# ------------------ PATHS ------------------
MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"
SELECTOR_PATH = BASE_DIR / "models" / "selector.pkl"
FEATURE_PATH = BASE_DIR / "models" / "raw_features.pkl"

REPORTS_PATH = BASE_DIR / "reports"
FIGURES_PATH = REPORTS_PATH / "figures"

# ------------------ TOP FEATURES (UI CLEAN) ------------------
TOP_FEATURES = [
    "mfcc_5_mean",
    "mfcc_2_mean",
    "mfcc_10_mean",
    "mfcc_12_mean",
    "mfcc_4_mean",
    "mean_pitch",
    "rms_energy",
    "zero_crossing_rate"
]

# ------------------ LOAD ------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    selector = joblib.load(SELECTOR_PATH)
    raw_features = joblib.load(FEATURE_PATH)
    return model, scaler, selector, raw_features

model, scaler, selector, raw_features = load_artifacts()

# ------------------ UI ------------------
st.set_page_config(page_title="Voice ML System", layout="wide")
st.title("🎙️ Human Voice Classification & Clustering")

st.sidebar.title("Navigation")
mode = st.sidebar.radio("Select Mode", ["Manual Input", "Upload CSV", "Clustering"])

# ================== MANUAL INPUT ==================
if mode == "Manual Input":

    st.header("🎛️ Enter Key Voice Features")

    inputs = {}
    col1, col2 = st.columns(2)

    for i, feature in enumerate(TOP_FEATURES):
        if i % 2 == 0:
            inputs[feature] = col1.number_input(feature, value=0.0)
        else:
            inputs[feature] = col2.number_input(feature, value=0.0)

    if st.button("🚀 Predict"):

        try:
            # Fill remaining features automatically
            full_input = {}

            for feature in raw_features:
                if feature in inputs:
                    full_input[feature] = inputs[feature]
                else:
                    full_input[feature] = 0  # default value

            input_df = pd.DataFrame([full_input])

            # Feature Engineering
            input_df = build_features(input_df)

            # Pipeline
            X_scaled = scaler.transform(input_df)
            X_selected = selector.transform(X_scaled)

            prediction = model.predict(X_selected)[0]
            result = "Male" if prediction == 1 else "Female"

            st.success(f"🎤 Prediction: {result}")

            # -------- Probability --------
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(X_selected)[0]

                st.subheader("Confidence Score")
                st.progress(int(max(prob) * 100))

                st.write({
                    "Female": round(prob[0], 3),
                    "Male": round(prob[1], 3)
                })

            # -------- Download --------
            output_df = input_df.copy()
            output_df["Prediction"] = result

            st.download_button(
                label="📥 Download Result",
                data=output_df.to_csv(index=False),
                file_name="prediction.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"Error: {e}")

# ================== CSV MODE ==================
elif mode == "Upload CSV":

    st.header("📂 Batch Prediction")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        try:
            df = df.drop(columns=["label"], errors="ignore")

            # Feature Engineering
            df = build_features(df)

            st.write("Preview", df.head())

            # Pipeline
            X_scaled = scaler.transform(df)
            X_selected = selector.transform(X_scaled)

            predictions = model.predict(X_selected)

            df["Prediction"] = ["Male" if p == 1 else "Female" for p in predictions]

            st.dataframe(df)

            # Chart
            st.subheader("Prediction Distribution")
            st.bar_chart(df["Prediction"].value_counts())

            # Download
            st.download_button(
                label="📥 Download Predictions",
                data=df.to_csv(index=False),
                file_name="batch_predictions.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"Error: {e}")

# ================== CLUSTERING ==================
elif mode == "Clustering":

    st.header("📊 Clustering Insights")

    st.metric("Best K", "2")
    st.metric("Silhouette Score", "0.295")

    elbow_path = FIGURES_PATH / "elbow_method.png"
    if elbow_path.exists():
        st.subheader("Elbow Method")
        st.image(str(elbow_path))

    kmeans_path = FIGURES_PATH / "KMeans_clusters.png"
    if kmeans_path.exists():
        st.subheader("KMeans Clusters")
        st.image(str(kmeans_path))