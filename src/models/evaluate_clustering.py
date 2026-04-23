from sklearn.metrics import silhouette_score
from src.utils.logger import get_logger

logger = get_logger(__name__)

def evaluate_clustering(model, X):
    labels = model.labels_

    unique_labels = set(labels)

    # Invalid clustering case
    if len(unique_labels) <= 1 or -1 in unique_labels:
        return -1, labels

    score = silhouette_score(X, labels)
    logger.info(f"Silhouette score: {score}")

    return score, labels