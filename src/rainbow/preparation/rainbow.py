"""Dataset preparation for rainbow datasets."""

import csv
import hashlib
import io
import json
import logging
import os
from typing import Dict, List, Optional
import zipfile

import tensorflow as tf

from .. import settings, utils
from . import preparer


logger = logging.getLogger(__name__)


# helper classes


@utils.data
class RainbowSplit:
    """A single split from a rainbow dataset for preprocessing."""

    name: str
    size: int

    features_path: str
    labels_path: Optional[str]


@utils.data
class RainbowArchive:
    """An archive of data for a rainbow dataset."""

    url: str
    checksum: str
    file_name: str

    splits: List[str]


@utils.data
class RainbowDataset:
    """A rainbow dataset for preprocessing."""

    name: str
    splits: Dict[str, RainbowSplit]

    archives: Dict[str, RainbowArchive]
    feature_names: List[str]


# main class


class RainbowPreparer(preparer.Preparer):
    """Prepare the rainbow datasets for text-to-text modeling."""

    RAINBOW_DATASETS = {
        # AlphaNLI
        "anli": RainbowDataset(
            name="anli",
            splits={
                "train": RainbowSplit(
                    name="train",
                    size=169654,
                    features_path="train.jsonl",
                    labels_path="train-labels.lst",
                ),
                "validation": RainbowSplit(
                    name="validation",
                    size=1532,
                    features_path="dev.jsonl",
                    labels_path="dev-labels.lst",
                ),
                "test": RainbowSplit(
                    name="test",
                    size=3040,
                    features_path="alphanli-test/anli.jsonl",
                    labels_path=None,
                ),
            },
            archives={
                "main": RainbowArchive(
                    url="gs://ai2-mosaic/public/alphanli/alphanli-train-dev.zip",
                    checksum="24840b27553e93ec625ae020dbf78d92daeae4be31ebbd469a0c9f6f99ed1c8d",
                    file_name="alphanli-train-dev.zip",
                    splits=["train", "validation"],
                ),
                "test": RainbowArchive(
                    url="gs://ai2-mosaic/public/alphanli/alphanli-test.zip",
                    checksum="afa4281fb78ca1081278ec19c53548f1c8ba599e863eebe84697aa6ce2e06e63",
                    file_name="alphanli-test.zip",
                    splits=["test"],
                ),
            },
            feature_names=["obs1", "obs2", "hyp1", "hyp2"],
        ),
        # CosmosQA
        "cosmosqa": RainbowDataset(
            name="cosmosqa",
            splits={
                "train": RainbowSplit(
                    name="train",
                    size=25262,
                    features_path="train.jsonl",
                    labels_path="train-labels.lst",
                ),
                "validation": RainbowSplit(
                    name="validation",
                    size=2985,
                    features_path="valid.jsonl",
                    labels_path="valid-labels.lst",
                ),
                "test": RainbowSplit(
                    name="test",
                    size=6963,
                    features_path="test.jsonl",
                    labels_path=None,
                ),
            },
            archives={
                "main": RainbowArchive(
                    url="gs://ai2-mosaic/public/cosmosqa/cosmosqa-data.zip",
                    checksum="d06bfef918240b34b6f86fdfd8215d15fea2abced0bfca8ab99004e3bce760ec",
                    file_name="cosmosqa-data.zip",
                    splits=["train", "validation", "test"],
                )
            },
            feature_names=[
                "context",
                "question",
                "answer0",
                "answer1",
                "answer2",
                "answer3",
            ],
        ),
        # HellaSWAG
        "hellaswag": RainbowDataset(
            name="hellaswag",
            splits={
                "train": RainbowSplit(
                    name="train",
                    size=39905,
                    features_path="hellaswag-train-dev/train.jsonl",
                    labels_path="hellaswag-train-dev/train-labels.lst",
                ),
                "validation": RainbowSplit(
                    name="validation",
                    size=10042,
                    features_path="hellaswag-train-dev/valid.jsonl",
                    labels_path="hellaswag-train-dev/valid-labels.lst",
                ),
                "test": RainbowSplit(
                    name="test",
                    size=10050,
                    features_path="hellaswag-test/hellaswag.jsonl",
                    labels_path=None,
                ),
            },
            archives={
                "main": RainbowArchive(
                    url="gs://ai2-mosaic/public/hellaswag/hellaswag-train-dev.zip",
                    checksum="5d5d70300eff7af886c184477bb076fbfa24336cb300c52c3b6e62644d14d928",
                    file_name="hellaswag-train-dev.zip",
                    splits=["train", "validation"],
                ),
                "test": RainbowArchive(
                    url="gs://ai2-mosaic/public/hellaswag/hellaswag-test.zip",
                    checksum="9fd5a8b0c06802e6aff6eb766842c4524c3db38f2bc708377adb1063267ec1b0",
                    file_name="hellaswag-test.zip",
                    splits=["test"],
                ),
            },
            feature_names=["ctx", "ending_options"],
        ),
        # PhysicalIQA
        "physicaliqa": RainbowDataset(
            name="physicaliqa",
            splits={
                "train": RainbowSplit(
                    name="train",
                    size=16113,
                    features_path="physicaliqa-train-dev/train.jsonl",
                    labels_path="physicaliqa-train-dev/train-labels.lst",
                ),
                "validation": RainbowSplit(
                    name="validation",
                    size=1838,
                    features_path="physicaliqa-train-dev/dev.jsonl",
                    labels_path="physicaliqa-train-dev/dev-labels.lst",
                ),
                "test": RainbowSplit(
                    name="test",
                    size=3446,
                    features_path="physicaliqa-test/physicaliqa.jsonl",
                    labels_path=None,
                ),
            },
            archives={
                "main": RainbowArchive(
                    url="gs://ai2-mosaic/public/physicaliqa/physicaliqa-train-dev.zip",
                    checksum="54d32a04f59a7e354396f321723c8d7ec35cc6b08506563d8d1ffcc15ce98ddd",
                    file_name="physicaliqa-train-dev.zip",
                    splits=["train", "validation"],
                ),
                "test": RainbowArchive(
                    url="gs://ai2-mosaic/public/physicaliqa/physicaliqa-test.zip",
                    checksum="b16301a506297842de5b1e2918200c3245daabc4de3c9007151e6098996651cf",
                    file_name="physicaliqa-test.zip",
                    splits=["test"],
                ),
            },
            feature_names=["goal", "sol1", "sol2"],
        ),
        # SocialIQA
        "socialiqa": RainbowDataset(
            name="socialiqa",
            splits={
                "train": RainbowSplit(
                    name="train",
                    size=33410,
                    features_path="socialiqa-train-dev/train.jsonl",
                    labels_path="socialiqa-train-dev/train-labels.lst",
                ),
                "validation": RainbowSplit(
                    name="validation",
                    size=1954,
                    features_path="socialiqa-train-dev/dev.jsonl",
                    labels_path="socialiqa-train-dev/dev-labels.lst",
                ),
                "test": RainbowSplit(
                    name="test",
                    size=2059,
                    features_path="socialiqa-test/socialiqa.jsonl",
                    labels_path=None,
                ),
            },
            archives={
                "main": RainbowArchive(
                    url="gs://ai2-mosaic/public/socialiqa/socialiqa-train-dev.zip",
                    checksum="ee073914a0fc33265cfbcfc50ec20df9b2e07809c3f420f599138d1f394ef5c3",
                    file_name="socialiqa-train-dev.zip",
                    splits=["train", "validation"],
                ),
                "test": RainbowArchive(
                    url="gs://ai2-mosaic/public/socialiqa/socialiqa-test.zip",
                    checksum="f1780632200ade1ec0e84af546aa0df2bcb9052f412afedec8d1fe03cfc8b326",
                    file_name="socialiqa-test.zip",
                    splits=["test"],
                ),
            },
            feature_names=[
                "context",
                "question",
                "answerA",
                "answerB",
                "answerC",
            ],
        ),
        # WinoGrande
        "winogrande": RainbowDataset(
            name="winogrande",
            splits={
                "train": RainbowSplit(
                    name="train",
                    size=40398,
                    features_path="winogrande_1.1/train_xl.jsonl",
                    labels_path="winogrande_1.1/train_xl-labels.lst",
                ),
                "validation": RainbowSplit(
                    name="validation",
                    size=1267,
                    features_path="winogrande_1.1/dev.jsonl",
                    labels_path="winogrande_1.1/dev-labels.lst",
                ),
                "test": RainbowSplit(
                    name="test",
                    size=1767,
                    features_path="winogrande_1.1/test.jsonl",
                    labels_path=None,
                ),
            },
            archives={
                "main": RainbowArchive(
                    url="gs://ai2-mosaic/public/winogrande/winogrande_1.1.zip",
                    checksum="d5699402a41de2b4e6b3c6e2a1e6b4faab2130543a12019a8c5d4f35ec502d47",
                    file_name="winogrande_1.1.zip",
                    splits=["train", "validation", "test"],
                )
            },
            feature_names=["sentence", "option1", "option2"],
        ),
    }
    """Configuration data for the Rainbow datasets."""

    def prepare(self, src: str, dst: str, force_download: bool = False) -> None:
        """See ``rainbow.preparation.preparer.Preparer``."""
        for dataset in self.RAINBOW_DATASETS.values():
            # Create the directory for saving the source files.
            tf.io.gfile.makedirs(os.path.join(src, dataset.name))
            # Create the directory for the dataset's prepared files.
            tf.io.gfile.makedirs(os.path.join(dst, dataset.name))

            for archive in dataset.archives.values():
                src_path = os.path.join(src, dataset.name, archive.file_name)

                # Copy the archive to src_path from the URL.
                if not tf.io.gfile.exists(src_path) or force_download:
                    logger.info(
                        f"Downloading {dataset.name}'s archive"
                        f" {archive.file_name} from {archive.url} to"
                        f" {src_path}."
                    )
                    tf.io.gfile.copy(
                        archive.url, src_path, overwrite=force_download
                    )

                with tf.io.gfile.GFile(src_path, "rb") as src_file:
                    # Verify the dataset file against its checksum.
                    sha256 = hashlib.sha256()
                    chunk = None
                    while chunk != b"":
                        # Read in 64KB at a time.
                        chunk = src_file.read(64 * 1024)
                        sha256.update(chunk)
                    checksum = sha256.hexdigest()
                    if checksum != archive.checksum:
                        raise IOError(
                            f"The archive, {archive.file_name}, for"
                            f" {dataset.name} did not have the expected"
                            f" checksum. Try running with force_download=True"
                            f" to redownload all files, or consider updating"
                            f" the datasets' checksums."
                        )
                    # Return to the beginning of the file.
                    src_file.seek(0)

                    # Prepare the archive's splits.
                    with zipfile.ZipFile(src_file, "r") as src_zip:
                        for split_name in archive.splits:
                            split = dataset.splits[split_name]

                            # Prepare and write out the split.
                            dst_path = os.path.join(
                                dst,
                                dataset.name,
                                settings.PREPROCESSED_SPLIT_FILE_NAME_TEMPLATE.format(
                                    split=split.name, dataset=dataset.name
                                ),
                            )
                            if split.labels_path is None:
                                labels_file = io.BytesIO(
                                    b"N/A\n"
                                    * (split.size + 1)
                                    # N.B. Add an extra line to the file so
                                    # that if the features file is too long,
                                    # the labels file won't automatically stop
                                    # the iteration at the right length below
                                    # (i.e., ``zip(features_file,
                                    # labels_file)``, that way we can detect
                                    # the expected size of the split being
                                    # incorrect.
                                )
                            else:
                                labels_file = src_zip.open(
                                    split.labels_path, "r"
                                )

                            with tf.io.gfile.GFile(
                                dst_path, "w"
                            ) as dst_file, src_zip.open(
                                split.features_path, "r"
                            ) as features_file, labels_file:
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
                                        if isinstance(
                                            feature[feature_name], str
                                        ):
                                            feature_strings.append(
                                                f"<{feature_name}>"
                                                f"{feature[feature_name]}"
                                                f"</{feature_name}>"
                                            )
                                        elif isinstance(
                                            feature[feature_name], list
                                        ):
                                            for (
                                                option_idx,
                                                option_text,
                                            ) in enumerate(
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
                                        "index": rows_written,
                                        "inputs": inputs,
                                        "targets": targets,
                                    }
                                    if i == 0:
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
                                raise ValueError(
                                    f"Expected to write {split.size} rows for the"
                                    f" {split.name} split of {dataset.name}, instead"
                                    f" {rows_written} were written."
                                )

                logger.info(
                    f"Finished preprocessing the {archive.file_name} archive."
                )

            logger.info(f"Finished preprocessing {dataset.name}.")

        logger.info("Finished preprocessing all rainbow datasets.")
