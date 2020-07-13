"""Dataset preparation for CycIC."""

import csv
import hashlib
import json
import logging
import os
import zipfile

import tensorflow as tf

from .. import settings
from . import preparer, utils


logger = logging.getLogger(__name__)


# main class


class CycICPreparer(preparer.Preparer):
    """Prepare CycIC for text-to-text modeling."""

    CYC = {
        "name": "cyc",
        "splits": {
            "train": {
                "name": "train",
                "size": 10678,
                "features_file_path": "CycIC-train-dev/cycic_training_questions.jsonl",
                "labels_file_path": "CycIC-train-dev/cycic_training_labels.jsonl",
            },
            "validation": {
                "name": "validation",
                "size": 1525,
                "features_file_path": "CycIC-train-dev/cycic_dev_questions.jsonl",
                "labels_file_path": "CycIC-train-dev/cycic_dev_labels.jsonl",
            },
            "test": {
                "name": "test",
                "size": 3051,
                "features_file_path": "CycIC-test/cycic.jsonl",
            },
        },
        "url": "https://storage.googleapis.com/ai2-mosaic/public/cycic/CycIC-train-dev.zip",
        "checksum": "c29cd76e5be956ee657cf7bd2d23295897dff0c020a3a10b9f0ad6ab77c693e6",
        "file_name": "CycIC-train-dev.zip",
        "test_url": "https://storage.googleapis.com/ai2-mosaic/public/cycic/CycIC-test.zip",
        "test_checksum": "ed5ec4220d42265d9e6e8b9779275b2af08b8e918cc794599db8b90088c0a952",
        "test_file_name": "CycIC-test.zip",
        "feature_names": [
            "question",
            "answer_option0",
            "answer_option1",
            "answer_option2",
            "answer_option3",
            "answer_option4",
        ],
    }
    """Configuration data for CycIC."""

    def prepare(self, src: str, dst: str, force_download: bool = False) -> None:
        """See ``rainbow.preparation.preparer.Preparer``."""
        # Make sure all data has been copied to src.

        # Create the directory for saving the source files.
        tf.io.gfile.makedirs(os.path.join(src, self.CYC["name"]))

        src_path = os.path.join(src, self.CYC["name"], self.CYC["file_name"])
        test_src_path = os.path.join(
            src, self.CYC["name"], self.CYC["test_file_name"]
        )

        # Copy the train and validation set to src_path from the URL.
        if not tf.io.gfile.exists(src_path) or force_download:
            logger.info(
                f"Downloading {self.CYC['name']}'s train and validation"
                f" data from {self.CYC['url']} to {src_path}."
            )
            utils.copy_url_to_gfile(self.CYC["url"], src_path)
        # Copy the test set to test_src_path from the URL.
        if not tf.io.gfile.exists(test_src_path) or force_download:
            logger.info(
                f"Downloading {self.CYC['name']}'s test data from"
                f" {self.CYC['test_url']} to {test_src_path}."
            )
            utils.copy_url_to_gfile(self.CYC["test_url"], test_src_path)

        # Preprocess train and validation.
        with tf.io.gfile.GFile(src_path, "rb") as src_file:
            # Verify the dataset file against its checksum.
            sha256 = hashlib.sha256()
            chunk = None
            while chunk != b"":
                # Read in 64KB at a time.
                chunk = src_file.read(64 * 1024)
                sha256.update(chunk)
            checksum = sha256.hexdigest()
            if checksum != self.CYC["checksum"]:
                raise IOError(
                    f"The file for {self.CYC['name']} train and"
                    f" validation data did not have the expected"
                    f" checksum. Try running with force_download=True"
                    f" to redownload all files, or consider updating"
                    f" the datasets' checksums."
                )
            # Return to the beginning of the file.
            src_file.seek(0)

            # Prepare CycIC's splits and write them to dst.
            with zipfile.ZipFile(src_file, "r") as src_zip:
                # Create the directory for the dataset's prepared files.
                tf.io.gfile.makedirs(os.path.join(dst, self.CYC["name"]))
                for split in self.CYC["splits"].values():
                    if split["name"] == "test":
                        # We'll handle the test split separately later,
                        # so skip it for now.
                        continue

                    # Prepare and write out the split.
                    dst_path = os.path.join(
                        dst,
                        self.CYC["name"],
                        settings.PREPROCESSED_SPLIT_FILE_NAME_TEMPLATE.format(
                            split=split["name"], dataset=self.CYC["name"],
                        ),
                    )
                    with tf.io.gfile.GFile(
                        dst_path, "w"
                    ) as dst_file, src_zip.open(
                        split["features_file_path"], "r"
                    ) as features_file, src_zip.open(
                        split["labels_file_path"], "r"
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
                            label = str(json.loads(ln2)["correct_answer"])

                            # Create the inputs and targets.
                            feature_strings = []
                            for feature_name in self.CYC["feature_names"]:
                                feature_strings.append(
                                    f"<{feature_name}>"
                                    f"{feature.get(feature_name, '')}"
                                    f"</{feature_name}>"
                                )

                            inputs = f"[{self.CYC['name']}]:\n" + "\n".join(
                                feature_strings
                            )
                            targets = label

                            row = {
                                "index": rows_written,
                                "inputs": inputs,
                                "targets": targets,
                            }
                            if i == 0:
                                logger.info(
                                    f"\n\n"
                                    f"Example {row['index']} from"
                                    f" {self.CYC['name']}'s {split['name']}"
                                    f" split:\n"
                                    f"inputs:\n"
                                    f"{row['inputs']}\n"
                                    f"targets:\n"
                                    f"{row['targets']}\n"
                                    f"\n"
                                )

                            # Write to the CSV.
                            writer.writerow(row)
                            rows_written += 1

                    if rows_written != split["size"]:
                        logger.error(
                            f"Expected to write {split['size']} rows for the"
                            f" {split['name']} split of {self.CYC['name']}, instead"
                            f" {rows_written} were written."
                        )

        # Preprocess test data.
        with tf.io.gfile.GFile(test_src_path, "rb") as test_src_file:
            # Verify the dataset file against its checksum.
            sha256 = hashlib.sha256()
            chunk = None
            while chunk != b"":
                # Read in 64KB at a time.
                chunk = test_src_file.read(64 * 1024)
                sha256.update(chunk)
            checksum = sha256.hexdigest()
            if checksum != self.CYC["test_checksum"]:
                raise IOError(
                    f"The file for {self.CYC['name']}'s test data did"
                    f" not have the expected checksum. Try running with"
                    f" force_download=True to redownload all files, or"
                    f" consider updating the datasets' checksums."
                )
            # Return to the beginning of the file.
            test_src_file.seek(0)

            # Prepare CycIC's test split and write it to dst.
            with zipfile.ZipFile(test_src_file, "r") as test_src_zip:
                split = self.CYC["splits"]["test"]
                # Prepare and write out the split.
                dst_path = os.path.join(
                    dst,
                    self.CYC["name"],
                    settings.PREPROCESSED_SPLIT_FILE_NAME_TEMPLATE.format(
                        split=split["name"], dataset=self.CYC["name"],
                    ),
                )
                with tf.io.gfile.GFile(
                    dst_path, "w"
                ) as dst_file, test_src_zip.open(
                    split["features_file_path"], "r"
                ) as features_file:
                    rows_written = 0

                    writer = csv.DictWriter(
                        dst_file,
                        fieldnames=["index", "inputs", "targets"],
                        dialect="unix",
                    )
                    writer.writeheader()

                    for i, ln in enumerate(features_file):
                        # Parse the data.
                        feature = json.loads(ln)

                        # Create the inputs and targets.
                        feature_strings = []
                        for feature_name in self.CYC["feature_names"]:
                            feature_strings.append(
                                f"<{feature_name}>"
                                f"{feature.get(feature_name, '')}"
                                f"</{feature_name}>"
                            )

                        inputs = f"[{self.CYC['name']}]:\n" + "\n".join(
                            feature_strings
                        )
                        targets = "N/A"

                        row = {
                            "index": rows_written,
                            "inputs": inputs,
                            "targets": targets,
                        }
                        if i == 0:
                            logger.info(
                                f"\n\n"
                                f"Example {row['index']} from"
                                f" {self.CYC['name']}'s {split['name']}"
                                f" split:\n"
                                f"inputs:\n"
                                f"{row['inputs']}\n"
                                f"targets:\n"
                                f"{row['targets']}\n"
                                f"\n"
                            )

                        # Write to the CSV.
                        writer.writerow(row)
                        rows_written += 1

                if rows_written != split["size"]:
                    logger.error(
                        f"Expected to write {split['size']} rows for the"
                        f" {split['name']} split of {self.CYC['name']}, instead"
                        f" {rows_written} were written."
                    )

        logger.info(f"Finished processing {self.CYC['name']}.")
