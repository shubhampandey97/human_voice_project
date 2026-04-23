from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

from src.utils.logger import get_logger

logger = get_logger(__name__)

def train_clustering_models(X):
    models = {}

    # Find best K
    best_k = 2
    best_score = -1
    best_kmeans = None

    for k in range(2, 6):
        logger.info(f"Training KMeans with k={k}")

        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X)

        # Handle invalid clustering
        if len(set(labels)) > 1:
            score = silhouette_score(X, labels)
            logger.info(f"k={k}, Silhouette Score={score}")

            if score > best_score:
                best_score = score
                best_k = k
                best_kmeans = kmeans

    logger.info(f"Best KMeans: k={best_k}, score={best_score}")

    models["KMeans"] = (best_kmeans, best_score, best_k)

    logger.info("Training DBSCAN model")

    dbscan = DBSCAN(eps=0.5, min_samples=5)
    # dbscan.fit(X)
    # models["DBSCAN"] = dbscan

    return models