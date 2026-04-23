import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ------------------ Paths ------------------
BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"
REPORTS_PATH = BASE_DIR / "reports" / "figures"

# ------------------ Load Model ------------------
@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

model, scaler = load_model()

# ------------------ UI Config ------------------
st.set_page_config(page_title="Human Voice Classification", layout="wide")

st.title("🎙️ Human Voice Classification & Clustering App")

# ------------------ Sidebar ------------------
st.sidebar.header("Navigation")
option = st.sidebar.radio("Go to", ["Prediction", "Clustering Insights"])

# ================== Prediction ==================
if option == "Prediction":
    st.header("🔍 Predict Voice Gender")

    st.write("Upload a CSV file with the same features used in training")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("### Uploaded Data", df.head())

        try:
            X_scaled = scaler.transform(df)
            predictions = model.predict(X_scaled)

            df["Prediction"] = predictions
            df["Prediction"] = df["Prediction"].map({0: "Female", 1: "Male"})

            st.success("Prediction Completed ✅")
            st.dataframe(df)

        except Exception as e:
            st.error(f"Error: {e}")

# ================== Clustering ==================
elif option == "Clustering Insights":
    st.header("📊 Clustering Insights")

    # ---- Elbow Method ----
    st.subheader("Elbow Method")
    elbow_path = REPORTS_PATH / "elbow_method.png"

    if elbow_path.exists():
        st.image(str(elbow_path))
    else:
        st.warning("Elbow plot not found")

    # ---- KMeans Plot ----
    st.subheader("KMeans Clusters")
    kmeans_path = REPORTS_PATH / "KMeans_clusters.png"

    if kmeans_path.exists():
        st.image(str(kmeans_path))
    else:
        st.warning("KMeans plot not found")

    # ---- Reports ----
    st.subheader("Reports")

    try:
        with open(BASE_DIR / "reports/KMeans_clustering_report.txt") as f:
            st.text(f.read())
    except:
        st.warning("KMeans report not found")

    try:
        with open(BASE_DIR / "reports/DBSCAN_clustering_report.txt") as f:
            st.text(f.read())
    except:
        st.warning("DBSCAN report not found")