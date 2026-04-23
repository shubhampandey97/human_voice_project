import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from src.utils.logger import get_logger

logger = get_logger(__name__)

def preprocess_data(df: pd.DataFrame):
    """
    Preprocess the dataset:
    - Handle missing values
    - Remove duplicates
    - Separate features and target
    - Scale numerical features

    Args:
        df (pd.DataFrame): Raw dataframe

    Returns:
        X_scaled (np.array): Scaled features
        y (pd.Series): Target variable
        scaler (StandardScaler): Fitted scaler
    """

    df = df.copy()

    # 🔹 Basic Cleaning
    logger.info("Starting preprocessing...")

    # Remove duplicates
    initial_shape = df.shape
    df.drop_duplicates(inplace=True)
    logger.info(f"Removed duplicates: {initial_shape} → {df.shape}")

    # Handle missing values
    missing = df.isnull().sum().sum()
    if missing > 0:
        logger.warning(f"Missing values found: {missing}")
        df = df.dropna()
        logger.info("Missing values dropped")

    # 🔹 Target Separation
    if "label" not in df.columns:
        raise ValueError("Target column 'label' not found in dataset")

    X = df.drop("label", axis=1)
    y = df["label"]

    # 🔹 Feature Scaling (IMPORTANT: only features, not target)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    logger.info("Scaling completed")

    return X_scaled, y, scaler