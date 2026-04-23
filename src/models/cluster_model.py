from sklearn.cluster import KMeans, DBSCAN
from src.utils.logger import get_logger

logger = get_logger(__name__)

def train_clustering_models(X):
    models = {}

    logger.info("Training KMeans model")
    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans.fit(X)
    models["KMeans"] = kmeans

    logger.info("Training DBSCAN model")
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    dbscan.fit(X)
    models["DBSCAN"] = dbscan

    return models