"""Dataset preparation for JOCI (JHU Ordinal Commonsense Inference)."""

import codecs
import csv
import hashlib
import logging
import os
import random
import zipfile

import tensorflow as tf

from .. import settings, utils as rainbow_utils
from . import preparer, utils as preparer_utils


logger = logging.getLogger(__name__)


# main class


class JOCIPreparer(preparer.Preparer):
    """Prepare JOCI for text-to-text modeling."""

    JOCI = {
        "name": "joci",
        "splits": {
            "train": {"name": "train", "size": 34092},
            "validation": {"name": "validation", "size": 2500},
        },
        "url": "http://decomp.io/projects/common-sense-inference/joci.zip",
        "checksum": "7812ddfa6e58d6bc8010dc88d9d2600cb8c559fc978201223012256f609017cb",
        "file_name": "joci.zip",
        "csv_path": "joci.csv",
    }
    """Configuration data for JOCI."""

    def prepare(self, src: str, dst: str, force_download: bool = False) -> None:
        """See ``rainbow.preparation.preparer.Preparer``."""
        # Create the directory for saving the source files.
        tf.io.gfile.makedirs(os.path.join(src, self.JOCI["name"]))

        # Create the directory for saving the prepared files.
        tf.io.gfile.makedirs(os.path.join(dst, self.JOCI["name"]))

        src_path = os.path.join(src, self.JOCI["name"], self.JOCI["file_name"])

        # Copy the dataset to src_path from the URL.
        if not tf.io.gfile.exists(src_path) or force_download:
            logger.info(
                f"Downloading {self.JOCI['name']} from {self.JOCI['url']}"
                f" to {src_path}."
            )
            preparer_utils.copy_url_to_gfile(self.JOCI["url"], src_path)

        with tf.io.gfile.GFile(src_path, "rb") as src_file:
            # Verify the dataset file against its checksum.
            sha256 = hashlib.sha256()
            chunk = None
            while chunk != b"":
                # Read in 64KB at a time.
                chunk = src_file.read(64 * 1024)
                sha256.update(chunk)
            checksum = sha256.hexdigest()
            if checksum != self.JOCI["checksum"]:
                raise IOError(
                    f"The file for {self.JOCI['name']} did not have the"
                    f" expected checksum. Try running with force_download=True"
                    f" to redownload all files, or consider updating the"
                    f" datasets' checksums."
                )
            # Return to the beginning of the file.
            src_file.seek(0)

            # Read the data from the JOCI file.
            with zipfile.ZipFile(src_file, "r") as src_zip:
                with src_zip.open(self.JOCI["csv_path"], "r") as joci_csv:
                    joci_csv = codecs.getreader("utf-8")(joci_csv)
                    reader = csv.DictReader(joci_csv)

                    data = [x for x in reader]

        # Prepare and write the splits to dst.

        # Shuffle and split the JOCI data.
        random_state = random.getstate()
        random.seed(rainbow_utils.string_to_seed(self.JOCI["name"]))
        random.shuffle(data)
        random.setstate(random_state)

        for split in self.JOCI["splits"].values():
            dst_path = os.path.join(
                dst,
                self.JOCI["name"],
                settings.PREPROCESSED_SPLIT_FILE_NAME_TEMPLATE.format(
                    split=split["name"], dataset=self.JOCI["name"]
                ),
            )

            with tf.io.gfile.GFile(dst_path, "w") as dst_file:
                rows_written = 0

                writer = csv.DictWriter(
                    dst_file,
                    fieldnames=["index", "inputs", "targets"],
                    dialect="unix",
                )
                writer.writeheader()

                split_data, data = data[: split["size"]], data[split["size"] :]
                for i, row_in in enumerate(split_data):
                    row_out = {
                        "index": rows_written,
                        "inputs": (
                            f"[{self.JOCI['name']}]:\n"
                            f"<context>{row_in['CONTEXT']}</context>\n"
                            f"<hypothesis>{row_in['HYPOTHESIS']}</hypothesis>"
                        ),
                        "targets": row_in["LABEL"],
                    }
                    if i == 0:
                        logger.info(
                            f"\n\n"
                            f"Example {row_out['index']} from"
                            f" {self.JOCI['name']}'s {split['name']} split:\n"
                            f"inputs:\n"
                            f"{row_out['inputs']}\n"
                            f"targets:\n"
                            f"{row_out['targets']}\n"
                            f"\n"
                        )

                    # Write to the CSV.
                    writer.writerow(row_out)
                    rows_written += 1

            if rows_written != split["size"]:
                logger.error(
                    f"Expected to write {split.size} rows for the"
                    f" {split['name']} split of {self.JOCI['name']}, instead"
                    f" {rows_written} were written."
                )

        logger.info(f"Finished processing JOCI.")
