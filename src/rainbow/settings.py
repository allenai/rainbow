"""Settings for rainbow."""

import os


# dataset preprocessing

PREPROCESSED_SPLIT_FILE_NAME_TEMPLATE = "{split}.{dataset}-preprocessed.csv"
"""The file name template for the preprocessed splits."""

DATASETS_DIR = os.environ["RAINBOW_DATASETS_DIR"]
"""The directory storing the rainbow datasets."""

PREPROCESSED_DATASETS_DIR = os.environ["RAINBOW_PREPROCESSED_DATASETS_DIR"]
"""The directory containing preprocessed splits for the rainbow datasets."""


# learning curve experiments

LEARNING_CURVE_SIZES = [
    None,
    1,
    2,
    4,
    7,
    14,
    26,
    49,
    92,
    175,
    334,
    635,
    1211,
    2309,
    4402,
    8392,
    16000,
]
"""The dataset sizes at which to evaluate the learning curves."""


# logging and output

LOG_FORMAT = "%(asctime)s:%(levelname)s:%(name)s: %(message)s"
"""The format string for logging."""

TQDM_KWARGS = {"ncols": 72, "leave": False}
"""Key-word arguments for tqdm progress bars."""
