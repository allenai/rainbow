"""Dataset preparation for CommonsenseQA."""

import codecs
import csv
import hashlib
import json
import logging
import os

import tensorflow as tf

from .. import settings
from . import preparer, utils


logger = logging.getLogger(__name__)


# main class


class CommonsenseQAPreparer(preparer.Preparer):
    """Prepare CommonsenseQA for text-to-text modeling."""

    COMMONSENSE_QA = {
        "name": "commonsenseqa",
        "splits": {
            "train": {
                "name": "train",
                "size": 9741,
                "url": "https://s3.amazonaws.com/commensenseqa/train_rand_split.jsonl",
                "checksum": "58ffa3c8472410e24b8c43f423d89c8a003d8284698a6ed7874355dedd09a2fb",
                "file_name": "train_rand_split.jsonl",
            },
            "validation": {
                "name": "validation",
                "size": 1221,
                "url": "https://s3.amazonaws.com/commensenseqa/dev_rand_split.jsonl",
                "checksum": "3210497fdaae614ac085d9eb873dd7f4d49b6f965a93adadc803e1229fd8a02a",
                "file_name": "dev_rand_split.jsonl",
            },
            "test": {
                "name": "test",
                "size": 1140,
                "url": "https://s3.amazonaws.com/commensenseqa/test_rand_split_no_answers.jsonl",
                "checksum": "b426896d71a9cd064cf01cfaf6e920817c51701ef66028883ac1af2e73ad5f29",
                "file_name": "test_rand_split.jsonl",
            },
        },
    }
    """Configuration data for CommonsenseQA."""

    def prepare(self, src: str, dst: str, force_download: bool = False) -> None:
        """See ``rainbow.preparation.preparer.Preparer``."""
        # Create the directory for saving the source files.
        tf.io.gfile.makedirs(os.path.join(src, self.COMMONSENSE_QA["name"]))

        # Create the directory for saving the prepared files.
        tf.io.gfile.makedirs(os.path.join(dst, self.COMMONSENSE_QA["name"]))

        # Prepare the splits.
        for split in self.COMMONSENSE_QA["splits"].values():
            src_path = os.path.join(
                src, self.COMMONSENSE_QA["name"], split["file_name"]
            )

            if not tf.io.gfile.exists(src_path) or force_download:
                logger.info(
                    f"Downloading {self.COMMONSENSE_QA['name']}'s"
                    f" {split['name']} split from {split['url']} to"
                    f" {src_path}."
                )
                utils.copy_url_to_gfile(split["url"], src_path)

            with tf.io.gfile.GFile(src_path, "rb") as src_file:
                # Verify the dataset file against its checksum.
                sha256 = hashlib.sha256()
                chunk = None
                while chunk != b"":
                    # Read in 64KB at a time.
                    chunk = src_file.read(64 * 1024)
                    sha256.update(chunk)
                checksum = sha256.hexdigest()
                if checksum != split["checksum"]:
                    raise IOError(
                        f"The {self.COMMONSENSE_QA['name']} file,"
                        f" {split['file_name']}, did not have the expected"
                        f" checksum. Try running with force_download=True"
                        f" to redownload all files, or consider updating"
                        f" the datasets' checksums."
                    )
                # Return to the beginning of the file.
                src_file.seek(0)

                # Decode src_file.
                src_file = codecs.getreader("utf-8")(src_file)

                dst_path = os.path.join(
                    dst,
                    self.COMMONSENSE_QA["name"],
                    settings.PREPROCESSED_SPLIT_FILE_NAME_TEMPLATE.format(
                        split=split["name"], dataset=self.COMMONSENSE_QA["name"]
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

                    for i, ln in enumerate(src_file):
                        row_in = json.loads(ln)

                        question = row_in["question"]["stem"]
                        choices = {
                            choice["label"]: choice["text"]
                            for choice in row_in["question"]["choices"]
                        }
                        label = row_in.get("answerKey", "N/A")

                        row_out = {
                            "index": rows_written,
                            "inputs": (
                                f"[{self.COMMONSENSE_QA['name']}]:\n"
                                f"<question>{question}</question>\n"
                                f"<answerA>{choices['A']}</answerA>\n"
                                f"<answerB>{choices['B']}</answerB>\n"
                                f"<answerC>{choices['C']}</answerC>\n"
                                f"<answerD>{choices['D']}</answerD>\n"
                                f"<answerE>{choices['E']}</answerE>"
                            ),
                            "targets": label,
                        }
                        if i == 0 and split["name"] != "test":
                            logger.info(
                                f"\n\n"
                                f"Example {row_out['index']} from"
                                f" {self.COMMONSENSE_QA['name']}'s"
                                f" {split['name']} split:\n"
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
                        f" {split['name']} split of"
                        f" {self.COMMONSENSE_QA['name']}, instead"
                        f" {rows_written} were written."
                    )

        logger.info("Finished processing CommonsenseQA.")
