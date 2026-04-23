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

def run_pipeline():
    logger.info("Pipeline started")

    # Set MLflow experiment
    mlflow.set_experiment("Human Voice Classification")
    
    data_path = BASE_DIR / "data" / "raw" / "vocal_gender_features_new.csv"
    models_path =  BASE_DIR / "models"
    models_path.mkdir(exist_ok=True)

    with mlflow.start_run(run_name="main_pipeline"):
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

        # mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("test_samples", len(X_test))

        # Train Models
        models = train_models(X_train, y_train)

        best_model = None
        best_score = 0
        best_model_name = ""

        for name, model in models.items():
            # Get best estimator
            best_estimator = model.best_estimator_

            acc, report = evaluate_model(best_estimator, X_test, y_test)

            logger.info(f"{name} Best Params: {model.best_params_}")
            logger.info(f"{name} Accuracy: {acc}")
            logger.info(f"{name} Report:\n{report}")

            # Log metrics
            # mlflow.log_params({f"{name}_{k}": v for k, v in model.best_params_.items()})
            for param, value in model.best_params_.items():
                mlflow.log_param(f"{name}_{param}", value)

            mlflow.log_metric(f"{name}_accuracy", acc)

            # Log report
            mlflow.log_text(report, f"{name}_classification_report.txt")

            reports_path = BASE_DIR / "reports"
            reports_path.mkdir(exist_ok=True)

            report_file = reports_path / f"{name}_classification_report.txt"

            with open(report_file, "w") as f:
                f.write(report)

            logger.info(f"{name} classification report saved at {report_file}")
            
            # Log model
            mlflow.sklearn.log_model(best_estimator, name=name)

            if acc > best_score:
                best_score = acc
                best_model = best_estimator
                best_model_name = name

        # Log best model details
        mlflow.log_metric("best_accuracy", best_score)
        mlflow.log_param("best_model", best_model_name)

        # os.makedirs("models", exist_ok=True)

        # joblib.dump(best_model, models_path / "best_model.pkl")
        with open(models_path / "model_info.txt", "w") as f:
            f.write(f"Best Model: {best_model_name}\nAccuracy: {best_score}")
        joblib.dump(scaler, models_path / "scaler.pkl")

        logger.info(f"Best model: {best_model_name} with accuracy {best_score}")
        logger.info("Best model saved")

    logger.info("Pipeline completed successfully")


if __name__ == "__main__":
    run_pipeline()