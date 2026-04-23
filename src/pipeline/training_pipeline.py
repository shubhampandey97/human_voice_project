from src.utils.logger import get_logger
import joblib
import os
from pathlib import Path
import pandas as pd
import mlflow
import mlflow.sklearn

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

    # Set MLflow experiment
    mlflow.set_experiment("Human Voice Classification")
    
    data_path = BASE_DIR / "data" / "raw" / "vocal_gender_features_new.csv"
    models_path =  BASE_DIR / "models"
    models_path.mkdir(exist_ok=True)

    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)

        logger.info(f"Loading data from {data_path}")
        df = load_data(str(data_path))

        # Preprocessing
        X, y, scaler = preprocess_data(df)

        processed_df = pd.DataFrame(X, columns=df.drop("label", axis=1).columns)
        processed_df["label"] = y.values

        processed_path = BASE_DIR / "data" / "processed"
        processed_path.mkdir(exist_ok=True)

        processed_file = processed_path / "vocal_gender_features_cleaned.csv"
        processed_df.to_csv(processed_file, index=False)

        logger.info(f"Processed data saved at {processed_file}")

        # Feature Engineering
        X = build_features(X)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Train Models
        models = train_models(X_train, y_train)

        best_model = None
        best_score = 0
        best_model_name = ""

        for name, model in models.items():

            acc, report = evaluate_model(model, X_test, y_test)

            logger.info(f"{name} Accuracy: {acc}")
            logger.info(f"{name} Report:\n{report}")

            # Log metrics
            mlflow.log_metric(f"{name}_accuracy", acc)

            # Log model
            mlflow.sklearn.log_model(model, artifact_path=name)

            if acc > best_score:
                best_score = acc
                best_model = model
                best_model_name = name

        # Log best model details
        mlflow.log_metric("best_accuracy", best_score)
        mlflow.log_param("best_model", best_model_name)

        os.makedirs("models", exist_ok=True)

        joblib.dump(best_model, models_path / "best_model.pkl")
        joblib.dump(scaler, models_path / "scaler.pkl")

        logger.info(f"Best model: {best_model_name} with accuracy {best_score}")
        logger.info("Best model saved")

    logger.info("Pipeline completed successfully")


if __name__ == "__main__":
    run_pipeline()