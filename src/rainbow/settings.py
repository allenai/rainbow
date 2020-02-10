"""Settings for rainbow."""

import os


# Dataset Preprocessing

PREPROCESSED_SPLIT_FILE_NAME_TEMPLATE = "{split}.{dataset}-preprocessed.csv"
"""The file name template for the preprocessed splits."""

DATASETS_DIR = os.environ["RAINBOW_DATASETS_DIR"]
"""The directory storing the rainbow datasets."""

PREPROCESSED_DATASETS_DIR = os.environ["RAINBOW_PREPROCESSED_DATASETS_DIR"]
"""The directory containing preprocessed splits for the rainbow datasets."""


# logging and output

LOG_FORMAT = "%(asctime)s:%(levelname)s:%(name)s: %(message)s"
"""The format string for logging."""

TQDM_KWARGS = {"ncols": 72, "leave": False}
"""Key-word arguments for tqdm progress bars."""
