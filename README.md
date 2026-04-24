# 🎙️ Human Voice Classification & Clustering

> 🚀 End-to-End Machine Learning System with Explainability (SHAP) & Interactive UI

---

## 🔥 Project Overview

This project is a **production-ready Machine Learning application** that classifies human voices (Male/Female) and performs clustering analysis on audio features.

It goes beyond basic ML by integrating:

* ✅ Advanced **feature engineering**
* ✅ Optimized **model training (SVM)**
* ✅ **Clustering (KMeans + PCA)**
* ✅ **Explainability using SHAP**
* ✅ Fully interactive **Streamlit Web App**

---

## 🎯 Key Features

### 🧠 Machine Learning

* Support Vector Machine (SVM) for classification
* Feature selection for dimensionality reduction
* Hyperparameter tuning for optimal performance

### 🔍 Explainability (SHAP)

* KernelExplainer for SVM
* Uses `decision_function` for meaningful insights
* Displays **top contributing features**
* Color-coded feature impact (positive/negative)

### 📊 Clustering Insights

* KMeans clustering
* Elbow Method visualization
* Silhouette Score evaluation

### 🎛️ Interactive UI (Streamlit)

* Manual feature input
* CSV batch predictions
* Real-time predictions + confidence score
* SHAP visual explanations
* Downloadable results

---

## 🏗️ Project Architecture

```
User Input / CSV
        ↓
Feature Engineering (build_features)
        ↓
Scaling (StandardScaler)
        ↓
Feature Selection (SelectKBest)
        ↓
Model (SVM)
        ↓
Prediction + Probability
        ↓
SHAP Explainability
```

---

## 📁 Project Structure

```
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── best_model.pkl
│   ├── scaler.pkl
│   ├── selector.pkl
│   ├── feature_names.pkl
│
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── pipeline/
│
├── streamlit_app/
│   └── app.py
│
├── reports/
│   ├── figures/
│   └── evaluation reports
│
└── README.md
```

---

## 🚀 How to Run

### 🔧 1. Clone Repository

```bash
git clone https://github.com/your-username/human_voice_project.git
cd human_voice_project
```

### 🔧 2. Create Environment

```bash
conda create -n voice_env python=3.9
conda activate voice_env
```

### 🔧 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 🔧 4. Run Streamlit App

```bash
streamlit run streamlit_app/app.py
```

---

## 📊 Model Performance

* 🎯 Accuracy: **~99%**
* 📈 Precision/Recall: Near perfect
* ⚡ Robust performance across test data

---

## 🔍 SHAP Explainability (Core Highlight)

This project uses SHAP to answer:

> ❓ *Why did the model predict Male/Female?*

### Key Implementation:

* Uses `KernelExplainer` (since SVM is non-linear)
* Uses `decision_function` instead of `predict_proba`
* Uses **proper background distribution** (fixes zero-impact issue)

### Output:

* Top influencing features
* Direction of impact (↑ / ↓)
* Visual bar charts

---

## ⚠️ Challenges Solved

| Problem                    | Solution                                        |
| -------------------------- | ----------------------------------------------- |
| SHAP returning all zeros   | Used proper background + decision_function      |
| Feature mismatch errors    | Fixed pipeline alignment                        |
| PCA interpretability issue | Separated clustering & classification pipelines |
| Streamlit import errors    | Fixed project structure & paths                 |

---

## 🧠 Key Learnings

* Importance of **pipeline consistency**
* Difference between **predict_proba vs decision_function**
* Handling **SHAP with non-linear models**
* Real-world **ML debugging skills**
* Building **production-ready ML apps**

---

## 🌍 Deployment

👉 Easily deployable on:

* Streamlit Cloud
* Render
* AWS / GCP

---

## 💼 Use Cases

* Voice-based gender classification
* Audio signal analysis
* Speech processing pipelines
* ML explainability demos

---

## 🧑‍💻 Author

**Shubham Pandey**
Data Scientist | ML Engineer

---

## ⭐ Why This Project Stands Out

✔ End-to-end ML pipeline
✔ Explainability (rare in beginner projects)
✔ Clean UI + real usability
✔ Strong debugging + engineering depth

---

## 🚀 Future Improvements

* Deep Learning (CNN on spectrograms)
* Real-time voice input (microphone)
* Deployment with API (FastAPI)
* Model monitoring

---

## 📌 Final Note

This project is not just about predictions —
it’s about **understanding the model’s decisions**.

> *"Good models predict. Great models explain."*
