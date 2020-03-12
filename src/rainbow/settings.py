"""Settings for rainbow."""

import math
import os

import t5


# dataset preprocessing

PREPROCESSED_SPLIT_FILE_NAME_TEMPLATE = "{split}.{dataset}-preprocessed.csv"
"""The file name template for the preprocessed splits."""

DATASETS_DIR = os.environ["RAINBOW_DATASETS_DIR"]
"""The directory storing the rainbow datasets."""

PREPROCESSED_DATASETS_DIR = os.environ["RAINBOW_PREPROCESSED_DATASETS_DIR"]
"""The directory containing preprocessed splits for the rainbow datasets."""


# tensorflow datasets configuration

TFDS_DATASETS_DIR = os.environ["RAINBOW_TFDS_DATASETS_DIR"]
"""The directory for storing the TFDS datasets."""
# Configure T5 to use TFDS_DATASETS_DIR.
t5.data.set_tfds_data_dir_override(TFDS_DATASETS_DIR)


# learning curve experiments

LEARNING_CURVE_SIZES = (
    [None]
    + [
        # exponentially space the first 1/2 of points
        math.ceil((16000 / 6) ** ((i + 1) / 7))
        for i in range(7)
    ][:-1]
    + [
        # linearly space the second 1/2 of points
        math.ceil(((i + 1) / 6) * 16000)
        for i in range(6)
    ]
)
"""The dataset sizes at which to evaluate the learning curves."""


# logging and output

LOG_FORMAT = "%(asctime)s:%(levelname)s:%(name)s: %(message)s"
"""The format string for logging."""

TQDM_KWARGS = {"ncols": 72, "leave": False}
"""Key-word arguments for tqdm progress bars."""
