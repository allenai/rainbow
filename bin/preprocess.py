#! /usr/bin/env python

"""Preprocess the rainbow datasets for text-to-text modeling."""

import csv
import hashlib
import json
import logging
import os
import zipfile

import click
import tensorflow as tf

from rainbow import datasets, settings, utils


logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--src",
    type=str,
    required=True,
    envvar="RAINBOW_DATASETS_DIR",
    help="The directory to which to download the rainbow datasets. Defaults to"
    " the RAINBOW_DATASETS_DIR environment variable.",
)
@click.option(
    "--dst",
    type=str,
    required=True,
    envvar="RAINBOW_PREPROCESSED_DATASETS_DIR",
    help="The directory to which to write the preprocessed dataset"
    " files. Defaults to the RAINBOW_PREPROCESSED_DATASETS_DIR environment"
    " variable.",
)
@click.option(
    "--force-download",
    is_flag=True,
    help="Force downloads of all the datasets, otherwise only missing datasets"
    " will be downloaded.",
)
def preprocess(src: str, dst: str, force_download: bool) -> None:
    """Preprocess the rainbow datasets for text-to-text modeling.

    Download to and read the rainbow datasets from --src, transform them into
    CSVs suitable for text-to-text models, then write the results to
    --dst. Google storage paths are supported.
    """
    utils.configure_logging()

    # Validate the arguments.

    if tf.io.gfile.exists(dst):
        raise IOError(f"Destination directory ({dst}) already exists.")

    # Download and preprocess the datasets.

    for dataset in datasets.RAINBOW_DATASETS.values():
        src_path = os.path.join(src, dataset.name, dataset.file_name)

        # Copy the dataset to src_path from the URL.
        if not tf.io.gfile.exists(src_path) or force_download:
            logger.info(
                f"Downloading {dataset.name} from {dataset.url} to {src_path}."
            )
            tf.io.gfile.copy(dataset.url, src_path, overwrite=force_download)

        with tf.io.gfile.GFile(src_path, "rb") as src_file:
            # Verify the dataset file against its checksum.
            sha256 = hashlib.sha256()
            chunk = None
            while chunk != b"":
                # Read in 64KB at a time.
                chunk = src_file.read(64 * 1024)
                sha256.update(chunk)
            checksum = sha256.hexdigest()
            if checksum != dataset.checksum:
                raise IOError(
                    f"The file for {dataset.name} did not have the expected"
                    f" checksum. Try running with --force-download to"
                    f" redownload all files, or consider updating the datasets'"
                    f" checksums."
                )
            # Return to the beginning of the file.
            src_file.seek(0)

            # Preprocess the dataset's splits.
            with zipfile.ZipFile(src_file, "r") as src_zip:
                for split in dataset.splits.values():
                    # Create the directory for the dataset's preprocessed files.
                    tf.io.gfile.makedirs(os.path.join(dst, dataset.name))
                    # Preprocess and write out the split.
                    dst_path = os.path.join(
                        dst,
                        dataset.name,
                        settings.PREPROCESSED_SPLIT_FILE_NAME_TEMPLATE.format(
                            split=split.name, dataset=dataset.name
                        ),
                    )
                    with tf.io.gfile.GFile(
                        dst_path, "w"
                    ) as dst_file, src_zip.open(
                        split.features_path, "r"
                    ) as features_file, src_zip.open(
                        split.labels_path, "r"
                    ) as labels_file:
                        rows_written = 0
                        writer = csv.DictWriter(
                            dst_file,
                            fieldnames=["index", "inputs", "targets"],
                            dialect="unix",
                        )
                        writer.writeheader()
                        for i, (ln1, ln2) in enumerate(
                            zip(features_file, labels_file)
                        ):
                            # Parse the data.
                            feature = json.loads(ln1)
                            label = ln2.decode().strip()

                            # Create the inputs and targets.
                            feature_strings = []
                            for feature_name in dataset.feature_names:
                                if isinstance(feature[feature_name], str):
                                    feature_strings.append(
                                        f"<{feature_name}>"
                                        f"{feature[feature_name]}"
                                        f"</{feature_name}>"
                                    )
                                elif isinstance(feature[feature_name], list):
                                    for option_idx, option_text in enumerate(
                                        feature[feature_name]
                                    ):
                                        feature_strings.append(
                                            f"<{feature_name} {option_idx}>"
                                            f"{option_text}"
                                            f"</{feature_name} {option_idx}>"
                                        )
                                else:
                                    raise ValueError(
                                        f"Unable to process feature of type"
                                        f" {feature[feature_name].__class__.__name__}."
                                    )

                            inputs = f"[{dataset.name}]:\n" + "\n".join(
                                feature_strings
                            )
                            targets = label

                            row = {
                                "index": i,
                                "inputs": inputs,
                                "targets": targets,
                            }
                            if i < 5:
                                logger.info(
                                    f"\n\n"
                                    f"Example {row['index']} from"
                                    f" {dataset.name}'s {split.name} split:\n"
                                    f"inputs:\n"
                                    f"{row['inputs']}\n"
                                    f"targets:\n"
                                    f"{row['targets']}\n"
                                    f"\n"
                                )

                            # Write to the CSV.
                            writer.writerow(row)
                            rows_written += 1

                    if rows_written != split.size:
                        logger.error(
                            f"Expected to write {split.size} rows for the"
                            f" {split.name} split of {dataset.name}, instead"
                            f" {rows_written} were written."
                        )

        logger.info(f"Finished preprocessing {dataset.name}.")

    logger.info("Finished preprocessing all datasets.")


if __name__ == "__main__":
    preprocess()  # pylint: disable=no-value-for-parameter
