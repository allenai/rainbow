"""Dataset preparation for ATOMIC."""

import codecs
import csv
import hashlib
import json
import logging
import os
import tarfile

import tensorflow as tf

from .. import settings
from . import preparer, utils


logger = logging.getLogger(__name__)


# main class


class AtomicPreparer(preparer.Preparer):
    """Prepare ATOMIC for text-to-text modeling."""

    ATOMIC = {
        "name": "atomic",
        "splits": {
            "train": {
                "name": "train",
                "size": 2 * 709996,
                "file_path": "v4_atomic_trn.csv",
            },
            "validation": {
                "name": "validation",
                "size": 2 * 79600,
                "file_path": "v4_atomic_dev.csv",
            },
        },
        "url": "https://homes.cs.washington.edu/~msap/atomic/data/atomic_data.tgz",
        "checksum": "2347ce175f68e42129828c07602299937c2a502390b9b2575fd10e1b78e076e3",
        "file_name": "atomic_data.tgz",
        "relations": [
            "oEffect",
            "oReact",
            "oWant",
            "xAttr",
            "xEffect",
            "xIntent",
            "xNeed",
            "xReact",
            "xWant",
        ],
    }
    """Configuration data for ATOMIC."""

    def prepare(self, src: str, dst: str, force_download: bool = False) -> None:
        """See ``rainbow.preparation.preparer.Preparer``."""
        # Make sure all data has been copied to src.

        # Create the directory for saving the source files.
        tf.io.gfile.makedirs(os.path.join(src, self.ATOMIC["name"]))

        src_path = os.path.join(
            src, self.ATOMIC["name"], self.ATOMIC["file_name"]
        )

        # Copy the dataset to src_path from the URL.
        if not tf.io.gfile.exists(src_path) or force_download:
            logger.info(
                f"Downloading {self.ATOMIC['name']} from {self.ATOMIC['url']}"
                f" to {src_path}."
            )
            utils.copy_url_to_gfile(self.ATOMIC["url"], src_path)

        with tf.io.gfile.GFile(src_path, "rb") as src_file:
            # Verify the dataset file against its checksum.
            sha256 = hashlib.sha256()
            chunk = None
            while chunk != b"":
                # Read in 64KB at a time.
                chunk = src_file.read(64 * 1024)
                sha256.update(chunk)
            checksum = sha256.hexdigest()
            if checksum != self.ATOMIC["checksum"]:
                raise IOError(
                    f"The file for {self.ATOMIC['name']} did not have the"
                    f" expected checksum. Try running with force_download=True"
                    f" to redownload all files, or consider updating the"
                    f" datasets' checksums."
                )
            # Return to the beginning of the file.
            src_file.seek(0)

            # Prepare ATOMIC's splits and write them to dst.

            with tarfile.open(fileobj=src_file, mode="r:gz") as src_tar:
                # Create the directory for ATOMIC's prepared files.
                tf.io.gfile.makedirs(os.path.join(dst, self.ATOMIC["name"]))
                for split in self.ATOMIC["splits"].values():
                    dst_path = os.path.join(
                        dst,
                        self.ATOMIC["name"],
                        settings.PREPROCESSED_SPLIT_FILE_NAME_TEMPLATE.format(
                            split=split["name"], dataset=self.ATOMIC["name"]
                        ),
                    )
                    with tf.io.gfile.GFile(
                        dst_path, "w"
                    ) as dst_file, src_tar.extractfile(
                        split["file_path"]
                    ) as split_file:
                        # Decode the split_file.
                        split_file = codecs.getreader("utf-8")(split_file)

                        rows_written = 0

                        writer = csv.DictWriter(
                            dst_file,
                            fieldnames=["index", "inputs", "targets"],
                            dialect="unix",
                        )
                        writer.writeheader()

                        reader = csv.DictReader(split_file)
                        for i, row_in in enumerate(reader):
                            for relation in self.ATOMIC["relations"]:
                                for object_ in json.loads(row_in[relation]):
                                    # create an example of KB completion in the
                                    # forward direction (subject, relation ->
                                    # object)
                                    row_out_forward = {
                                        "index": rows_written,
                                        "inputs": (
                                            f"[{self.ATOMIC['name']}]:\n"
                                            f"<subject>{row_in['event']}</subject>\n"
                                            f"<relation>{relation}</relation>"
                                        ),
                                        "targets": f"<object>{object_}</object>",
                                    }

                                    # create an example of KB completion in the
                                    # backward direction (object, relation ->
                                    # subject)
                                    row_out_backward = {
                                        "index": rows_written + 1,
                                        "inputs": (
                                            f"[{self.ATOMIC['name']}]:\n"
                                            f"<object>{object_}</object>\n"
                                            f"<relation>{relation}</relation>"
                                        ),
                                        "targets": f"<subject>{row_in['event']}</subject>",
                                    }

                                    if i == 0:
                                        logger.info(
                                            f"\n\n"
                                            f"Example {row_out_forward['index']} from"
                                            f" {self.ATOMIC['name']}'s"
                                            f" {split['name']} split:\n"
                                            f"inputs:\n"
                                            f"{row_out_forward['inputs']}\n"
                                            f"targets:\n"
                                            f"{row_out_forward['targets']}\n"
                                            f"\n"
                                        )
                                        logger.info(
                                            f"\n\n"
                                            f"Example {row_out_backward['index']} from"
                                            f" {self.ATOMIC['name']}'s"
                                            f" {split['name']} split:\n"
                                            f"inputs:\n"
                                            f"{row_out_backward['inputs']}\n"
                                            f"targets:\n"
                                            f"{row_out_backward['targets']}\n"
                                            f"\n"
                                        )

                                    # Write to the CSV.
                                    writer.writerow(row_out_forward)
                                    writer.writerow(row_out_backward)
                                    rows_written += 2

                    if rows_written != split["size"]:
                        logger.error(
                            f"Expected to write {split['size']} rows for the"
                            f" {split['name']} split of {self.ATOMIC['name']},"
                            f" instead {rows_written} were written."
                        )

        logger.info("Finished preparing ATOMIC.")
