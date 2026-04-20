from src.utils.logger import get_logger
import joblib
import os
from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split

from src.data.load_data import load_data
from src.data.preprocess import preprocess_data
from src.features.build_features import build_features
from src.models.train_model import train_models
from src.models.evaluate import evaluate_model

logger = get_logger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
print("BASE_DIR:", BASE_DIR)

def run_pipeline():
    logger.info("Pipeline started")
    
    data_path = BASE_DIR / "data" / "raw" / "vocal_gender_features_new.csv"
    print("DATA PATH:", data_path)
    models_path =  BASE_DIR / "models"
    models_path.mkdir(exist_ok=True)

    df = load_data(str(data_path))

    X, y, scaler = preprocess_data(df)

    processed_df = pd.DataFrame(X, columns=df.drop("label", axis=1).columns)
    processed_df["label"] = y.values

    processed_path = BASE_DIR / "data" / "processed"
    processed_path.mkdir(exist_ok=True)

    processed_file = processed_path / "vocal_gender_features_cleaned.csv"

    processed_df.to_csv(processed_file, index=False)

    logger.info(f"Processed data saved at {processed_file}")

    X = build_features(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = train_models(X_train, y_train)

    best_model = None
    best_score = 0

    for name, model in models.items():

        acc, report = evaluate_model(model, X_test, y_test)

        logger.info(f"{name} Accuracy: {acc}")

        if acc > best_score:
            best_score = acc
            best_model = model

    os.makedirs("models", exist_ok=True)

    joblib.dump(best_model, models_path / "best_model.pkl")
    joblib.dump(scaler, models_path / "scaler.pkl")

    logger.info("Best model saved")
    logger.info("Pipeline completed successfully")


if __name__ == "__main__":
    run_pipeline()