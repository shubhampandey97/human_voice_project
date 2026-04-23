import pandas as pd
import mlflow
import mlflow.sklearn
from pathlib import Path

from src.data.load_data import load_data
from src.data.preprocess import preprocess_data
from src.models.cluster_model import train_clustering_models
from src.models.evaluate_clustering import evaluate_clustering
from src.utils.logger import get_logger

logger = get_logger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]


def run_clustering_pipeline():
    logger.info("Clustering pipeline started")

    mlflow.set_experiment("Voice Clustering")

    data_path = BASE_DIR / "data" / "raw" / "vocal_gender_features_new.csv"
    logger.info(f"Loading data from {data_path}")

    with mlflow.start_run(run_name="clustering_pipeline"):

        # 🔹 Load Data
        df = load_data(str(data_path))
        logger.info(f"Dataset shape: {df.shape}")

        # 🔹 Preprocess
        X, y, scaler = preprocess_data(df)
        logger.info("Preprocessing completed")

        # 🔹 Train Clustering Models
        logger.info("Training clustering models")
        models = train_clustering_models(X)

        # 🔹 Evaluate
        for name, model in models.items():

            logger.info(f"Evaluating {name}")

            score = evaluate_clustering(model, X)

            logger.info(f"{name} Silhouette Score: {score}")

            # MLflow logging
            mlflow.log_metric(f"{name}_silhouette_score", score)
            mlflow.sklearn.log_model(model, name=name)

        logger.info("All clustering models evaluated")

    logger.info("Clustering pipeline completed successfully")


if __name__ == "__main__":
    run_clustering_pipeline()