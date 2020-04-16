#! /usr/bin/env python

"""Prepare all relevant datasets for text-to-text modeling."""

import logging

import click
import tensorflow as tf

from rainbow import preparation, utils


logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--src",
    type=str,
    required=True,
    envvar="RAINBOW_DATASETS_DIR",
    help=(
        "The directory to which to download all the relevant"
        " datasets. Defaults to the RAINBOW_DATASETS_DIR environment"
        " variable."
    ),
)
@click.option(
    "--dst",
    type=str,
    required=True,
    envvar="RAINBOW_PREPROCESSED_DATASETS_DIR",
    help=(
        "The directory to which to write the preprocessed dataset"
        " files. Defaults to the RAINBOW_PREPROCESSED_DATASETS_DIR environment"
        " variable."
    ),
)
@click.option(
    "--force-download",
    is_flag=True,
    help=(
        "Force downloads of all the datasets, otherwise only missing datasets"
        " will be downloaded."
    ),
)
def prepare(src: str, dst: str, force_download: bool) -> None:
    """Prepare all relevant datasets for text-to-text modeling.

    Download to and read the datasets from --src, transform them into
    CSVs suitable for text-to-text models, then write the results to
    --dst. Google storage paths are supported.
    """
    utils.configure_logging(clear=True)

    # Validate the arguments.

    if tf.io.gfile.exists(dst):
        raise IOError(f"Destination directory ({dst}) already exists.")

    # Download and preprocess the datasets.

    preparation.rainbow.RainbowPreparer().prepare(
        src=src, dst=dst, force_download=force_download
    )

    preparation.atomic.AtomicPreparer().prepare(
        src=src, dst=dst, force_download=force_download
    )

    preparation.conceptnet.ConceptNetPreparer().prepare(
        src=src, dst=dst, force_download=force_download
    )

    preparation.commonsenseqa.CommonsenseQAPreparer().prepare(
        src=src, dst=dst, force_download=force_download
    )

    preparation.joci.JOCIPreparer().prepare(
        src=src, dst=dst, force_download=force_download
    )

    logger.info(f"All datasets have been prepared.")


if __name__ == "__main__":
    prepare()  # pylint: disable=no-value-for-parameter
