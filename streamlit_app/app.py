import sys
from pathlib import Path

# ------------------ PATH FIX ------------------
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))

import streamlit as st
import pandas as pd
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt

from src.features.build_features import build_features

# ------------------ PATHS ------------------
MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"
SELECTOR_PATH = BASE_DIR / "models" / "selector.pkl"
RAW_FEATURE_PATH = BASE_DIR / "models" / "raw_features.pkl"
SELECTED_FEATURE_PATH = BASE_DIR / "models" / "feature_names.pkl"

REPORTS_PATH = BASE_DIR / "reports"
FIGURES_PATH = REPORTS_PATH / "figures"

# ------------------ TOP FEATURES (UI) ------------------
TOP_FEATURES = [
    "mfcc_5_mean", "mfcc_2_mean", "mfcc_10_mean",
    "mfcc_12_mean", "mfcc_4_mean", "mean_pitch",
    "rms_energy", "zero_crossing_rate"
]

# ------------------ LOAD ARTIFACTS ------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    selector = joblib.load(SELECTOR_PATH)
    raw_features = joblib.load(RAW_FEATURE_PATH)
    selected_features = joblib.load(SELECTED_FEATURE_PATH)
    return model, scaler, selector, raw_features, selected_features

model, scaler, selector, raw_features, selected_features = load_artifacts()

# ------------------ SHAP FUNCTION ------------------
def plot_shap(model, X_selected, feature_names):
    try:
        # Background (important fix)
        if X_selected.shape[0] > 1:
            idx = np.random.choice(X_selected.shape[0], min(50, X_selected.shape[0]), replace=False)
            background = X_selected[idx]
        else:
            background = np.random.normal(0, 1, size=(20, X_selected.shape[1]))

        explainer = shap.KernelExplainer(model.decision_function, background)
        shap_values = explainer.shap_values(X_selected)

        values = np.array(shap_values).reshape(-1)

        # Align features
        min_len = min(len(feature_names), len(values))
        values = values[:min_len]
        feature_names = feature_names[:min_len]

        shap_df = pd.DataFrame({
            "feature": feature_names,
            "impact": values
        })

        shap_df["abs"] = shap_df["impact"].abs()
        shap_df = shap_df.sort_values(by="abs", ascending=False).drop(columns=["abs"])

        # Top 10
        shap_df = shap_df.head(10)

        # Highlight top feature
        top_feat = shap_df.iloc[0]
        st.info(f"Top driver: {top_feat['feature']} ({top_feat['impact']:.3f})")

        # Colored chart
        colors = ["green" if v > 0 else "red" for v in shap_df["impact"]]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(shap_df["feature"], shap_df["impact"], color=colors)
        ax.invert_yaxis()
        ax.set_title("Feature Impact (SHAP)")
        ax.set_xlabel("Impact")

        st.pyplot(fig)

    except Exception as e:
        st.error(f"SHAP failed: {e}")

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
            # Fill full feature set
            full_input = {f: inputs.get(f, 0) for f in raw_features}

            input_df = pd.DataFrame([full_input])

            # Pipeline
            input_df = build_features(input_df)
            X_scaled = scaler.transform(input_df)
            X_selected = selector.transform(X_scaled)

            prediction = model.predict(X_selected)[0]
            result = "Male" if prediction == 1 else "Female"

            st.success(f"🎤 Prediction: {result}")

            # SHAP
            st.subheader("🔍 Model Explanation")
            plot_shap(model, X_selected, selected_features)

            # Probability
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(X_selected)[0]
                st.subheader("Confidence Score")
                st.progress(int(max(prob) * 100))
                st.write({"Female": round(prob[0], 3), "Male": round(prob[1], 3)})

            # Download
            input_df["Prediction"] = result
            st.download_button(
                "📥 Download Result",
                input_df.to_csv(index=False),
                "prediction.csv",
                "text/csv"
            )

        except Exception as e:
            st.error(f"Error: {e}")

# ================== CSV MODE ==================
elif mode == "Upload CSV":

    st.header("📂 Batch Prediction")

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        df = pd.read_csv(file)

        try:
            df = df.drop(columns=["label"], errors="ignore")

            df = build_features(df)

            X_scaled = scaler.transform(df)
            X_selected = selector.transform(X_scaled)

            predictions = model.predict(X_selected)

            df["Prediction"] = ["Male" if p == 1 else "Female" for p in predictions]

            st.dataframe(df)

            st.subheader("📊 Prediction Distribution")
            st.bar_chart(df["Prediction"].value_counts())

            # SHAP (first sample)
            st.subheader("🔍 Model Explanation (Sample)")
            plot_shap(model, X_selected[0:1], selected_features)

            st.download_button(
                "📥 Download Predictions",
                df.to_csv(index=False),
                "batch_predictions.csv",
                "text/csv"
            )

        except Exception as e:
            st.error(f"Error: {e}")

# ================== CLUSTERING ==================
elif mode == "Clustering":

    st.header("📊 Clustering Insights")

    st.metric("Best K", "2")
    st.metric("Silhouette Score", "0.295")

    elbow = FIGURES_PATH / "elbow_method.png"
    if elbow.exists():
        st.subheader("Elbow Method")
        st.image(str(elbow))

    cluster = FIGURES_PATH / "KMeans_clusters.png"
    if cluster.exists():
        st.subheader("Cluster Visualization")
        st.image(str(cluster))