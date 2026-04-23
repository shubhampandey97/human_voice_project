from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

from src.utils.logger import get_logger

logger = get_logger(__name__)

def train_models(X_train, y_train):

    logger.info("Training models started")

    models = {}
     # Random Forest Tuning
    rf_params = {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5]
    }

    rf = RandomForestClassifier(random_state=42)

    rf_grid = GridSearchCV(
        rf,
        rf_params,
        cv=3,
        scoring="accuracy",
        n_jobs=-1
    )

    rf_grid.fit(X_train, y_train)
    logger.info("Random Forest trained")

    models["RandomForest"] = rf_grid

    # SVM Tuning
    svm_params = {
        "C": [0.1, 1, 10],
        "kernel": ["linear", "rbf"]
    }

    svm = SVC(probability=True)

    svm_grid = GridSearchCV(
        svm,
        svm_params,
        cv=3,
        scoring="accuracy",
        n_jobs=-1
    )

    svm_grid.fit(X_train, y_train)
    logger.info("SVM trained")

    models["SVM"] = svm_grid

    logger.info("All models trained")

    return models