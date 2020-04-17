"""Dataset preparation for ReCoRD."""

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


class ReCoRDPreparer(preparer.Preparer):
    """Prepare ReCoRD for text-to-text modeling."""

    RECORD = {
        "name": "record",
        "splits": {
            "train": {
                "name": "train",
                "size": 257863,
                "file_path": "ReCoRD/train.jsonl",
            },
            "validation": {
                "name": "validation",
                "size": 29949,
                "file_path": "ReCoRD/val.jsonl",
            },
        },
        "url": "https://dl.fbaipublicfiles.com/glue/superglue/data/v2/ReCoRD.zip",
        "checksum": "30c7b651ab21b8bf8fab986495cd1084333010e040548f861b839eec0044ac18",
        "file_name": "ReCoRD.zip",
    }
    """Configuration data for ReCoRD."""

    def prepare(self, src: str, dst: str, force_download: bool = False) -> None:
        """See ``rainbow.preparation.preparer.Preparer``."""
        # Create the directory for saving the source files.
        tf.io.gfile.makedirs(os.path.join(src, self.RECORD["name"]))

        # Create the directory for saving the prepared files.
        tf.io.gfile.makedirs(os.path.join(dst, self.RECORD["name"]))

        src_path = os.path.join(
            src, self.RECORD["name"], self.RECORD["file_name"]
        )

        # Copy the dataset to src_path from the URL.
        if not tf.io.gfile.exists(src_path) or force_download:
            logger.info(
                f"Downloading {self.RECORD['name']} from {self.RECORD['url']}"
                f" to {src_path}."
            )
            utils.copy_url_to_gfile(self.RECORD["url"], src_path)

        with tf.io.gfile.GFile(src_path, "rb") as src_file:
            # Verify the dataset file against its checksum.
            sha256 = hashlib.sha256()
            chunk = None
            while chunk != b"":
                # Read in 64KB at a time.
                chunk = src_file.read(64 * 1024)
                sha256.update(chunk)
            checksum = sha256.hexdigest()
            if checksum != self.RECORD["checksum"]:
                raise IOError(
                    f"The file for {self.RECORD['name']} did not have the"
                    f" expected checksum. Try running with force_download=True"
                    f" to redownload all files, or consider updating the"
                    f" datasets' checksums."
                )
            # Return to the beginning of the file.
            src_file.seek(0)

            with zipfile.ZipFile(src_file, "r") as src_zip:
                # Prepare and write splits to dst.
                for split in self.RECORD["splits"].values():
                    dst_path = os.path.join(
                        dst,
                        self.RECORD["name"],
                        settings.PREPROCESSED_SPLIT_FILE_NAME_TEMPLATE.format(
                            split=split["name"], dataset=self.RECORD["name"]
                        ),
                    )

                    with tf.io.gfile.GFile(
                        dst_path, "w"
                    ) as dst_file, src_zip.open(
                        split["file_path"], "r"
                    ) as split_json:
                        rows_written = 0

                        writer = csv.DictWriter(
                            dst_file,
                            fieldnames=["index", "inputs", "targets"],
                            dialect="unix",
                        )
                        writer.writeheader()

                        for i, ln in enumerate(split_json):
                            row_in = json.loads(ln)

                            passage = row_in["passage"]["text"]
                            entities = "\n".join(
                                set(
                                    f"<entity>{passage[e['start']:e['end'] + 1]}</entity>"
                                    for e in row_in["passage"]["entities"]
                                )
                            )
                            qas = row_in["qas"]
                            for qa in qas:
                                query = qa["query"]
                                for answer in qa["answers"]:
                                    row_out = {
                                        "index": rows_written,
                                        "inputs": (
                                            f"[{self.RECORD['name']}]:\n"
                                            f"<query>{query}</query>\n"
                                            f"{entities}\n"
                                            f"<passage>{passage}</passage>"
                                        ),
                                        "targets": answer["text"],
                                    }
                                    if i == 0:
                                        logger.info(
                                            f"\n\n"
                                            f"Example {row_out['index']} from"
                                            f" {self.RECORD['name']}'s {split['name']} split:\n"
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
                            f"Expected to write {split['size']} rows for the"
                            f" {split['name']} split of {self.RECORD['name']}, instead"
                            f" {rows_written} were written."
                        )

        logger.info(f"Finished processing ReCoRD.")
