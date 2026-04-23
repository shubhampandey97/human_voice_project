from sklearn.metrics import silhouette_score
from src.utils.logger import get_logger

logger = get_logger(__name__)

def evaluate_clustering(model, X):
    labels = model.labels_

    if len(set(labels)) > 1 and -1 not in set(labels):
        score = silhouette_score(X, labels)
        logger.info(f"Silhouette score: {score}")
    else:
        score = -1
        logger.warning("Invalid clustering for silhouette score")

    return score