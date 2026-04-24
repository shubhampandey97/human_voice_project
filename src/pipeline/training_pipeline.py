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

import shap
import matplotlib.pyplot as plt

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
        raw_features = df.drop("label", axis=1).columns

        # Feature Engineering
        df = build_features(df)
        logger.info(f"Features after engineering: {df.shape}")

        # Preprocessing
        X, y, scaler, selector, selected_features = preprocess_data(df)

        # Save processed data
        # processed_df = pd.DataFrame(X, columns=df.drop("label", axis=1).columns)
        processed_df = pd.DataFrame(X, columns=selected_features)
        processed_df["label"] = y.values

        processed_path = BASE_DIR / "data" / "processed"
        processed_path.mkdir(exist_ok=True)

        processed_file = processed_path / "vocal_gender_features_cleaned.csv"
        processed_df.to_csv(processed_file, index=False)

        logger.info(f"Processed data saved at {processed_file}")


        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("test_samples", len(X_test))

        # Train Models
        models = train_models(X_train, y_train)

        best_model = None
        best_score = 0
        best_model_name = ""

        reports_path = BASE_DIR / "reports"
        reports_path.mkdir(exist_ok=True)

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

        joblib.dump(best_model, models_path / "best_model.pkl")
        joblib.dump(scaler, models_path / "scaler.pkl")
        joblib.dump(selector, models_path / "selector.pkl")

        joblib.dump(raw_features, models_path / "raw_features.pkl")

        logger.info(f"Raw features saved at {models_path / 'raw_features.pkl'}")
                        
        with open(models_path / "model_info.txt", "w") as f:
            f.write(f"Best Model: {best_model_name}\nAccuracy: {best_score}")

        # ------------------ Save Scaler Report ------------------
        feature_names = df.drop("label", axis=1).columns

        scaler_df = pd.DataFrame({
            "feature": feature_names,
            "mean": scaler.mean_,
            "std_dev": scaler.scale_
        })

        scaler_csv_path = models_path / "scaler_info.csv"
        scaler_df.to_csv(scaler_csv_path, index=False)

        mlflow.log_artifact(str(scaler_csv_path), artifact_path="scaler")

        logger.info(f"Scaler report saved at {scaler_csv_path}")

        # Save as formatted TXT
        scaler_txt_path = models_path / "scaler_info.txt"

        with open(scaler_txt_path, "w") as f:
            f.write("=== SCALER REPORT ===\n\n")
            f.write("Type: StandardScaler\n")
            f.write(f"Total Features: {len(feature_names)}\n\n")

            f.write("{:<30} {:>15} {:>15}\n".format("Feature", "Mean", "Std Dev"))
            f.write("=" * 65 + "\n")

            for name, mean, std in zip(feature_names, scaler.mean_, scaler.scale_):
                f.write("{:<30} {:>15.6f} {:>15.6f}\n".format(name, mean, std))

        logger.info(f"Scaler report saved at {scaler_csv_path} and {scaler_txt_path}")

        with open(models_path / "selected_features.txt", "w") as f:
            for feat in selected_features:
                f.write(f"{feat}\n")

        logger.info(f"Best model: {best_model_name} with accuracy {best_score}")
        logger.info("Best model saved")

        logger.info("Generating SHAP explanations (using RandomForest)")

        # Find RandomForest model
        rf_model = None
        for name, model in models.items():
            if "RandomForest" in name:
                rf_model = model.best_estimator_

        # Use RF for SHAP (tree-based = fast + accurate)
        if rf_model is not None:

            X_sample = X_train[:200]

            explainer = shap.TreeExplainer(rf_model)
            shap_values = explainer.shap_values(X_sample)

            # Plot
            shap.summary_plot(shap_values, X_sample, show=False)

            reports_path = BASE_DIR / "reports"
            reports_path.mkdir(exist_ok=True)

            shap_file = reports_path / "shap_summary.png"
            plt.savefig(shap_file)
            plt.close()

            logger.info(f"SHAP plot saved at {shap_file}")

            mlflow.log_artifact(str(shap_file))

        else:
            logger.warning("RandomForest model not found for SHAP")

    logger.info("Pipeline completed successfully")


if __name__ == "__main__":
    run_pipeline()