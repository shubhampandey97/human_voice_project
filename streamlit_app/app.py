import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ------------------ Paths ------------------
BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"
REPORTS_PATH = BASE_DIR / "reports"
FIGURES_PATH = REPORTS_PATH / "figures"

# ------------------ Load Model ------------------
@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

model, scaler = load_model()

# ------------------ Config ------------------
st.set_page_config(page_title="Voice Classification", layout="wide")

st.title("🎙️ Human Voice Classification & Clustering")
st.markdown("### 🚀 End-to-End ML Project | Classification + Clustering + MLflow")

# ------------------ Sidebar ------------------
st.sidebar.header("Navigation")
option = st.sidebar.radio("Select Module", ["Prediction", "Clustering Insights", "Model Info"])

# ================== Prediction ==================
if option == "Prediction":

    st.header("🔍 Voice Gender Prediction")

    tab1, tab2 = st.tabs(["📂 Upload CSV", "✍️ Manual Input"])

    # ---------- Upload CSV ----------
    with tab1:
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.write("### Input Data", df.head())

            try:
                X_scaled = scaler.transform(df)
                predictions = model.predict(X_scaled)

                df["Prediction"] = ["Male" if p == 1 else "Female" for p in predictions]

                st.success("Prediction Completed ✅")
                st.dataframe(df)

                # 📊 Metrics
                col1, col2 = st.columns(2)
                col1.metric("Total Samples", len(df))
                col2.metric("Male Count", (df["Prediction"] == "Male").sum())

                # 📥 Download
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Download Predictions",
                    csv,
                    "predictions.csv",
                    "text/csv"
                )

            except Exception as e:
                st.error(f"Error: {e}")

    # ---------- Manual Input ----------
    with tab2:
        st.subheader("Enter Features")

        feature_values = {}
        cols = st.columns(3)

        # Replace with your real feature names if needed
        for i in range(10):  # simplified UI (can expand to all features)
            with cols[i % 3]:
                feature_values[f"feature_{i}"] = st.number_input(f"Feature {i}", value=0.0)

        if st.button("Predict"):
            try:
                input_df = pd.DataFrame([feature_values])
                input_scaled = scaler.transform(input_df)
                pred = model.predict(input_scaled)[0]

                label = "👨 Male" if pred == 1 else "👩 Female"
                st.success(f"Prediction: {label}")

            except Exception as e:
                st.error(f"Error: {e}")

# ================== Clustering ==================
elif option == "Clustering Insights":

    st.header("📊 Clustering Analysis")

    col1, col2 = st.columns(2)

    # Elbow
    with col1:
        st.subheader("Elbow Method")
        elbow_path = FIGURES_PATH / "elbow_method.png"
        if elbow_path.exists():
            st.image(str(elbow_path))
        else:
            st.warning("Elbow plot not found")

    # KMeans
    with col2:
        st.subheader("KMeans Clusters")
        kmeans_path = FIGURES_PATH / "KMeans_clusters.png"
        if kmeans_path.exists():
            st.image(str(kmeans_path))
        else:
            st.warning("KMeans plot not found")

    # Insights
    st.subheader("📌 Key Insights")
    st.info("KMeans shows weak clustering (low silhouette score)")
    st.warning("DBSCAN could not find meaningful clusters")

    # Reports
    with st.expander("📄 View Reports"):
        try:
            with open(REPORTS_PATH / "KMeans_clustering_report.txt") as f:
                st.text(f.read())
        except:
            st.warning("KMeans report missing")

        try:
            with open(REPORTS_PATH / "DBSCAN_clustering_report.txt") as f:
                st.text(f.read())
        except:
            st.warning("DBSCAN report missing")

# ================== Model Info ==================
elif option == "Model Info":

    st.header("📌 Model Details")

    st.markdown("""
    ### 🔍 Model Summary
    - Model Used: **Support Vector Machine (SVM)**
    - Accuracy: **~99.9%**
    - Dataset: Voice acoustic features

    ### ⚙️ Pipeline
    - Data Preprocessing
    - Feature Scaling
    - Model Training
    - Hyperparameter Tuning (GridSearchCV)
    - MLflow Tracking

    ### 📊 Clustering
    - KMeans (Elbow Method)
    - DBSCAN
    - PCA Visualization

    ### 🚀 Deployment
    - Streamlit App
    """)

    st.success("This is a production-level ML pipeline 🚀")