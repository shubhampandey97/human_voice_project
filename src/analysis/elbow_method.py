from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from pathlib import Path
import mlflow

from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_elbow_method(X, reports_path):
    logger.info("Running Elbow Method")

    inertia = []

    K = range(1, 10)

    for k in K:
        logger.info(f"Training KMeans with k={k}")
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        inertia.append(kmeans.inertia_)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(K, inertia, marker="o")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia")
    plt.title("Elbow Method")

    # Save plot
    figures_path = reports_path / "figures"
    figures_path.mkdir(exist_ok=True)

    elbow_file = figures_path / "elbow_method.png"
    plt.savefig(elbow_file)
    plt.close()

    logger.info(f"Elbow plot saved at {elbow_file}")

    # Log to MLflow
    mlflow.log_artifact(str(elbow_file))

    logger.info("Elbow method completed")