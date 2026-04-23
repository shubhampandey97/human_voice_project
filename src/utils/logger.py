import logging
import os
import sys

def get_logger(name: str):
    os.makedirs("logs", exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Prevent duplicate handlers
    if not logger.handlers:

        # File Handler (UTF-8 safe)
        file_handler = logging.FileHandler(
            "logs/pipeline.log", encoding="utf-8"
        )

        # Console Handler (UTF-8 safe)
        console_handler = logging.StreamHandler(sys.stdout)

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger