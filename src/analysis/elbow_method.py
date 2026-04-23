from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from src.utils.logger import get_logger

logger = get_logger(__name__)

def run_elbow_method(X):
    logger.info("Running Elbow Method")

    inertia = []

    for k in range(1, 10):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        inertia.append(kmeans.inertia_)

    plt.plot(range(1, 10), inertia)
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia")
    plt.title("Elbow Method")
    plt.show()

    logger.info("Elbow method completed")