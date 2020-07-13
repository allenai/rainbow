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
        },
        "url": "https://storage.googleapis.com/ai2-mosaic/public/cycic/CycIC-train-dev.zip",
        "checksum": "c29cd76e5be956ee657cf7bd2d23295897dff0c020a3a10b9f0ad6ab77c693e6",
        "file_name": "CycIC-train-dev.zip",
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

        # Copy the dataset to src_path from the URL.
        if not tf.io.gfile.exists(src_path) or force_download:
            logger.info(
                f"Downloading {self.CYC['name']} from {self.CYC['url']}"
                f" to {src_path}."
            )
            utils.copy_url_to_gfile(self.CYC["url"], src_path)

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
                    f"The file for {self.CYC['name']} did not have the"
                    f" expected checksum. Try running with force_download=True"
                    f" to redownload all files, or consider updating the"
                    f" datasets' checksums."
                )
            # Return to the beginning of the file.
            src_file.seek(0)

            # Prepare CycIC's splits and write them to dst.
            with zipfile.ZipFile(src_file, "r") as src_zip:
                # Create the directory for the dataset's prepared files.
                tf.io.gfile.makedirs(os.path.join(dst, self.CYC["name"]))
                for split in self.CYC["splits"].values():
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

            logger.info(f"Finished processing {self.CYC['name']}.")
