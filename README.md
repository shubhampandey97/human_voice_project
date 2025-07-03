## ✅ Final Folder Structure

```
human_voice_project/
├── data/
│   └── vocal_gender_features_cleaned.csv         ← cleaned dataset
├── models/
│   ├── best_model.pkl                             ← trained classifier
│   └── scaler.pkl                                 ← StandardScaler
├── notebooks/
│   ├── 01_eda_and_preprocessing.ipynb             ← EDA
│   ├── 02_clustering_models.ipynb                 ← KMeans, DBSCAN
│   └── 03_classification_models.ipynb             ← ML classifiers
├── streamlit_app/
│   └── app.py                                     ← Streamlit UI
├── utils/                                         ← (optional helper scripts)
├── requirements.txt                               ← dependencies
├── README.md                                      ← project guide
└── .gitignore                                     ← exclude .pkl, __pycache__
```

```markdown
# 🎙️ Human Voice Classification and Clustering

This project applies machine learning to classify and cluster human voices (male/female) based on extracted audio features.

## 📦 Features
- EDA with spectral, pitch, and MFCC analysis
- Clustering with KMeans and DBSCAN
- Classification using Logistic Regression, Random Forest, SVM, Neural Network
- Streamlit web app for real-time predictions
- Saved trained model and scaler for deployment

## 📁 Folder Structure
```

human_voice_project/
├── data/
├── models/
├── notebooks/
├── streamlit_app/
├── utils/
├── requirements.txt
└── README.md
````

## 🚀 Streamlit App
Launch the app:
```bash
cd streamlit_app
streamlit run app.py
````

Use either:

* Manual feature input
* Upload 1-row CSV sample

## 🛠️ Requirements

```bash
pip install -r requirements.txt
```

## 📊 Sample Features

43 voice features including:

* Spectral centroid, bandwidth, contrast, flatness
* Pitch (mean, max, min)
* 13 MFCC mean and std coefficients

## 📈 Model Evaluation

Models trained and evaluated using:

* Accuracy, F1-score, Confusion Matrix (classification)
* Silhouette Score (clustering)

## 👨‍💻 Author

Shubham Pandey
