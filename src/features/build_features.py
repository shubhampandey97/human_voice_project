import numpy as np
from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_features(df):
    logger.info("Feature engineering started")

    # Pitch-Energy Interaction
    df["pitch_energy_ratio"] = df["mean_pitch"] / (df["rms_energy"] + 1e-6)

    # Spectral Shape
    df["spectral_shape"] = (
        df["mean_spectral_centroid"]
        * df["mean_spectral_bandwidth"]
        / (df["mean_spectral_rolloff"] + 1e-6)
    )

    # MFCC Mean Aggregation
    mfcc_mean_cols = [c for c in df.columns if "mfcc" in c and "mean" in c]
    df["mfcc_mean_avg"] = df[mfcc_mean_cols].mean(axis=1)

    # MFCC Variability
    mfcc_std_cols = [c for c in df.columns if "mfcc" in c and "std" in c]
    df["mfcc_variability"] = df[mfcc_std_cols].mean(axis=1)

    # Pitch Range
    df["pitch_range"] = df["max_pitch"] - df["min_pitch"]

    logger.info("Feature engineering completed")

    return df