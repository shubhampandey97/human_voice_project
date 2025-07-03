# import streamlit as st
# import pandas as pd
# import joblib

# # Load model and scaler
# model = joblib.load('../models/best_model.pkl')
# scaler = joblib.load('../models/scaler.pkl')

# # Streamlit UI
# st.title("🎙️ Human Voice Gender Classification")
# st.write("Upload a voice feature row to predict gender.")

# uploaded_file = st.file_uploader("Upload CSV File with Feature Row (1 sample)", type="csv")

# if uploaded_file is not None:
#     input_df = pd.read_csv(uploaded_file)
    
#     # Check structure
#     if input_df.shape[1] != 43:
#         st.error("Expected 43 input features.")
#     else:
#         st.write("✅ Input Preview:", input_df.head())

#         # Scale input
#         input_scaled = scaler.transform(input_df)

#         # Predict
#         prediction = model.predict(input_scaled)[0]
#         label = "Male" if prediction == 1 else "Female"

#         st.success(f"Predicted Gender: **{label}**")


import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load saved model and scaler
model = joblib.load('../models/best_model.pkl')
scaler = joblib.load('../models/scaler.pkl')

# App Title
st.set_page_config(page_title="Voice Gender Classification", layout="centered")
st.title("🎙️ Human Voice Gender Classification")
st.markdown("Upload a voice feature CSV or enter values manually to classify voice as **Male or Female**.")

# Sidebar navigation
st.sidebar.title("🔍 Prediction Options")
option = st.sidebar.radio("Choose input method:", ["Manual Input", "Upload CSV"])

# Expected 43 feature names (same order as training)
FEATURES = [
    'mean_spectral_centroid', 'std_spectral_centroid', 'mean_spectral_bandwidth', 'std_spectral_bandwidth',
    'mean_spectral_contrast', 'mean_spectral_flatness', 'mean_spectral_rolloff', 'zero_crossing_rate',
    'rms_energy', 'mean_pitch', 'min_pitch', 'max_pitch', 'std_pitch', 'spectral_skew', 'spectral_kurtosis',
    'energy_entropy', 'log_energy'
] + [f'mfcc_{i}_{stat}' for i in range(1, 14) for stat in ['mean', 'std']]

# Manual Input UI
if option == "Manual Input":
    st.subheader("🔧 Enter Voice Features")
    manual_input = {}

    # Split into 3 columns
    col1, col2, col3 = st.columns(3)
    for idx, feature in enumerate(FEATURES):
        with [col1, col2, col3][idx % 3]:
            manual_input[feature] = st.number_input(f"{feature}", format="%.4f", step=0.01)

    if st.button("Predict"):
        input_df = pd.DataFrame([manual_input])
        scaled_input = scaler.transform(input_df)
        prediction = model.predict(scaled_input)[0]
        label = "👩 Female" if prediction == 0 else "👨 Male"
        st.success(f"### 🎯 Predicted Gender: {label}")

# Upload CSV UI
if option == "Upload CSV":
    st.subheader("📂 Upload CSV File")
    uploaded_file = st.file_uploader("Upload a CSV file containing 1 row of 43 features", type=["csv"])
    
    if uploaded_file:
        try:
            input_df = pd.read_csv(uploaded_file)
            if input_df.shape[1] != 43:
                st.error("❌ CSV must contain exactly 43 columns.")
            else:
                st.dataframe(input_df)
                scaled_input = scaler.transform(input_df)
                prediction = model.predict(scaled_input)[0]
                label = "👩 Female" if prediction == 0 else "👨 Male"
                st.success(f"### 🎯 Predicted Gender: {label}")
        except Exception as e:
            st.error(f"Error processing file: {e}")
