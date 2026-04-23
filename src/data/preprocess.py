import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from src.utils.logger import get_logger

logger = get_logger(__name__)

def remove_outliers_iqr(df, factor=1.5):
    """Clip outliers using IQR"""
    numeric_cols = df.select_dtypes(include=np.number).columns.drop("label")

    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - factor * IQR
        upper = Q3 + factor * IQR

        df.loc[:, col] = np.clip(df[col], lower, upper)

    return df

def preprocess_data(df):
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

    # Basic Cleaning
    logger.info("Starting preprocessing...")

    # Remove duplicates
    before = df.shape
    df = df.drop_duplicates()
    after = df.shape
    logger.info(f"Removed duplicates: {before} → {after}")

    # Outlier handling
    df = remove_outliers_iqr(df)
    logger.info("Outliers handled using IQR")

    # Handle inf and NaN values
    df = df.replace([np.inf, -np.inf], 0)
    df = df.fillna(0)

    # Handle missing values
    missing = df.isnull().sum().sum()
    if missing > 0:
        logger.warning(f"Missing values found: {missing}")
        df = df.dropna()
        logger.info("Missing values dropped")

    # Target Separation
    if "label" not in df.columns:
        raise ValueError("Target column 'label' not found in dataset")

    X = df.drop("label", axis=1)
    y = df["label"]

    # Feature Scaling (IMPORTANT: only features, not target)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    logger.info("Scaling completed")

    return X_scaled, y, scaler