import pandas as pd
import os
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load dataset from a given path.

    Args:
        file_path (str): Path to the CSV file

    Returns:
        pd.DataFrame: Loaded dataframe
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ File not found at: {file_path}")

    try:
        df = pd.read_csv(file_path)
        logger.info(f"✅ Data loaded successfully from {file_path}")
        logger.info(f"📊 Shape of dataset: {df.shape}")
    except Exception as e:
        raise RuntimeError(f"Error loading data: {e}")

    return df