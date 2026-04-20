from sklearn.metrics import accuracy_score, classification_report
from src.utils.logger import get_logger

logger = get_logger(__name__)

def evaluate_model(model, X_test, y_test):

    logger.info("Evaluating model")

    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds)

    logger.info(f"Model accuracy: {acc:.4f}")

    return acc, report