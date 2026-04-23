import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from pathlib import Path

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from src.data.load_data import load_data
from src.data.preprocess import preprocess_data
from src.models.cluster_model import train_clustering_models
from src.models.evaluate_clustering import evaluate_clustering
from src.utils.logger import get_logger
from src.analysis.elbow_method import run_elbow_method

from sklearn.preprocessing import StandardScaler


logger = get_logger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]


def run_clustering_pipeline():
    logger.info("Clustering pipeline started")

    mlflow.set_experiment("Voice Clustering")

    data_path = BASE_DIR / "data" / "raw" / "vocal_gender_features_new.csv"
    logger.info(f"Loading data from {data_path}")

    with mlflow.start_run(run_name="clustering_pipeline"):

        # Load Data
        df = load_data(str(data_path))
        logger.info(f"Dataset shape: {df.shape}")

        # Preprocess
        X, y, scaler, selector, feature_names = preprocess_data(df)

        logger.info("Preprocessing completed")

        mlflow.log_text(str(feature_names), "selected_features.txt")

        models_path = BASE_DIR / "models"
        models_path.mkdir(exist_ok=True)

        # Save selected features for app usage
        import joblib
        joblib.dump(feature_names, models_path / "feature_names.pkl")

        # Save readable version
        with open(models_path / "selected_features.txt", "w") as f:
            for feat in feature_names:
                f.write(f"{feat}\n")

        print("Saving feature_names at:", models_path / "feature_names.pkl")

        logger.info("Applying PCA for dimensionality reduction")

        pca_model = PCA(n_components=5)
        X = pca_model.fit_transform(X)

        mlflow.log_param("pca_components", 5)

        reports_path = BASE_DIR / "reports"
        reports_path.mkdir(exist_ok=True)

        # elbow method
        run_elbow_method(X, reports_path)

        # Train Models
        logger.info("Training clustering models")
        models = train_clustering_models(X)

        figures_path = reports_path / "figures"
        figures_path.mkdir(exist_ok=True)

        # Evaluate
        for name, model_data in models.items():

            # Handle both simple & tuned models
            if isinstance(model_data, tuple):
                model, preset_score, best_k = model_data
            else:
                model = model_data
                preset_score = None
                best_k = None

            logger.info(f"Evaluating {name}")

            with mlflow.start_run(run_name=name, nested=True):

                score, labels = evaluate_clustering(model, X)

                logger.info(f"{name} Silhouette Score: {score}")

                # Log metric
                mlflow.log_metric("silhouette_score", score)

                # Log model name
                mlflow.set_tag("model_name", name)

                # Log best_k (only for KMeans)
                if best_k:
                    mlflow.log_param("best_k", best_k)

                # Cluster distribution
                unique, counts = np.unique(labels, return_counts=True)

                distribution_text = ""
                for u, c in zip(unique, counts):
                    line = f"Cluster {u}: {c} samples"
                    logger.info(line)
                    distribution_text += line + "\n"

                    # Log cluster counts as metrics
                    mlflow.log_metric(f"cluster_{u}_count", int(c))

                # Save report file
                report_file = reports_path / f"{name}_clustering_report.txt"

                with open(report_file, "w") as f:
                    f.write(f"Model: {name}\n")
                    f.write(f"Silhouette Score: {score}\n\n")
                    f.write("Cluster Distribution:\n")
                    f.write(distribution_text)

                    if best_k:
                        f.write(f"\nBest K (KMeans): {best_k}\n")

                logger.info(f"{name} clustering report saved at {report_file}")

                # Log report (ONLY ONCE)
                mlflow.log_artifact(str(report_file))

                # Save labels (NEW)
                labels_df = pd.DataFrame({"cluster": labels})
                labels_file = reports_path / f"{name}_labels.csv"
                labels_df.to_csv(labels_file, index=False)

                mlflow.log_artifact(str(labels_file))

                # PCA VISUALIZATION (only if valid clustering)
                if len(set(labels)) > 1:
                    pca_vis = PCA(n_components=2)
                    X_pca = pca_vis.fit_transform(X)

                    plt.figure(figsize=(8, 6))
                    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels)
                    plt.title(f"{name} Clustering")

                    plot_file = figures_path / f"{name}_clusters.png"
                    plt.savefig(plot_file)
                    plt.close()

                    logger.info(f"{name} cluster plot saved at {plot_file}")

                    mlflow.log_artifact(str(plot_file))
                else:
                    logger.warning(f"{name} has invalid clustering. Skipping visualization.")

                # Log model
                mlflow.sklearn.log_model(model, name=name)

        logger.info("All clustering models evaluated")

    logger.info("Clustering pipeline completed successfully")


if __name__ == "__main__":
    run_clustering_pipeline()