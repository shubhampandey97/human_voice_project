from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from src.utils.logger import get_logger

logger = get_logger(__name__)

def train_models(X_train, y_train):

    logger.info("Training models started")

    models = {}

    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    logger.info("Random Forest trained")

    svm = SVC(probability=True)
    svm.fit(X_train, y_train)
    logger.info("SVM trained")

    models["RandomForest"] = rf
    models["SVM"] = svm

    logger.info("All models trained")

    return models