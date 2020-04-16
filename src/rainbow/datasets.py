"""Dataset definitions for rainbow."""

from typing import Dict, List

import t5

from . import utils


# N.B. The names and sizes for the datasets must correspond to the names
# and sizes used during dataset preparation (see
# $REPO/bin/prepare.py and $REPO/src/rainbow/preparation/).


# core classes


@utils.data
class Split:
    """A single split of a dataset."""

    name: str
    size: int


@utils.data
class Dataset:
    """A dataset."""

    name: str
    splits: Dict[str, Split]


# constants

RAINBOW_DATASETS = {
    # AlphaNLI
    "anli": Dataset(
        name="anli",
        splits={
            "train": Split(name="train", size=169654),
            "validation": Split(name="validation", size=1532),
        },
    ),
    # CosmosQA
    "cosmosqa": Dataset(
        name="cosmosqa",
        splits={
            "train": Split(name="train", size=25262),
            "validation": Split(name="validation", size=2985),
        },
    ),
    # HellaSWAG
    "hellaswag": Dataset(
        name="hellaswag",
        splits={
            "train": Split(name="train", size=39905),
            "validation": Split(name="validation", size=10042),
        },
    ),
    # PhysicalIQA
    "physicaliqa": Dataset(
        name="physicaliqa",
        splits={
            "train": Split(name="train", size=16113),
            "validation": Split(name="validation", size=1838),
        },
    ),
    # SocialIQA
    "socialiqa": Dataset(
        name="socialiqa",
        splits={
            "train": Split(name="train", size=33410),
            "validation": Split(name="validation", size=1954),
        },
    ),
    # WinoGrande
    "winogrande": Dataset(
        name="winogrande",
        splits={
            "train": Split(name="train", size=40398),
            "validation": Split(name="validation", size=1267),
        },
    ),
}
"""Rainbow datasets."""


KNOWLEDGE_GRAPH_DATASETS = {
    # ATOMIC
    "atomic": Dataset(
        name="atomic",
        splits={
            "train": Split(name="train", size=2 * 709996),
            "validation": Split(name="validation", size=2 * 79600),
        },
    ),
    # ConceptNet
    "conceptnet": Dataset(
        name="conceptnet",
        splits={
            "train": Split(name="train", size=2 * 100000),
            "validation": Split(name="validation", size=2 * 1200),
        },
    ),
}
"""Commonsense knowledge graph datasets."""


GLUE_DATASETS = {
    # CoLA
    "cola": Dataset(
        name="cola",
        splits={
            "train": Split(
                name="train",
                size=t5.data.get_mixture_or_task(
                    "glue_cola_v002"
                ).num_input_examples("train"),
            ),
            "validation": Split(
                name="validation",
                size=t5.data.get_mixture_or_task(
                    "glue_cola_v002"
                ).num_input_examples("validation"),
            ),
        },
    ),
    # SST2
    "sst2": Dataset(
        name="sst2",
        splits={
            "train": Split(
                name="train",
                size=t5.data.get_mixture_or_task(
                    "glue_sst2_v002"
                ).num_input_examples("train"),
            ),
            "validation": Split(
                name="validation",
                size=t5.data.get_mixture_or_task(
                    "glue_sst2_v002"
                ).num_input_examples("validation"),
            ),
        },
    ),
    # MRPC
    "mrpc": Dataset(
        name="mrpc",
        splits={
            "train": Split(
                name="train",
                size=t5.data.get_mixture_or_task(
                    "glue_mrpc_v002"
                ).num_input_examples("train"),
            ),
            "validation": Split(
                name="validation",
                size=t5.data.get_mixture_or_task(
                    "glue_mrpc_v002"
                ).num_input_examples("validation"),
            ),
        },
    ),
    # STS-B
    "stsb": Dataset(
        name="stsb",
        splits={
            "train": Split(
                name="train",
                size=t5.data.get_mixture_or_task(
                    "glue_stsb_v002"
                ).num_input_examples("train"),
            ),
            "validation": Split(
                name="validation",
                size=t5.data.get_mixture_or_task(
                    "glue_stsb_v002"
                ).num_input_examples("validation"),
            ),
        },
    ),
    # QQP
    "qqp": Dataset(
        name="qqp",
        splits={
            "train": Split(
                name="train",
                size=t5.data.get_mixture_or_task(
                    "glue_qqp_v002"
                ).num_input_examples("train"),
            ),
            "validation": Split(
                name="validation",
                size=t5.data.get_mixture_or_task(
                    "glue_qqp_v002"
                ).num_input_examples("validation"),
            ),
        },
    ),
    # MNLI
    "mnli": Dataset(
        name="mnli",
        splits={
            "train": Split(
                name="train",
                size=t5.data.get_mixture_or_task(
                    "glue_mnli_v002"
                ).num_input_examples("train"),
            ),
            "validation": Split(
                name="validation_mismatched",
                size=t5.data.get_mixture_or_task(
                    "glue_mnli_v002"
                ).num_input_examples("validation_mismatched"),
            ),
        },
    ),
    # QNLI
    "qnli": Dataset(
        name="qnli",
        splits={
            "train": Split(
                name="train",
                size=t5.data.get_mixture_or_task(
                    "glue_qnli_v002"
                ).num_input_examples("train"),
            ),
            "validation": Split(
                name="validation",
                size=t5.data.get_mixture_or_task(
                    "glue_qnli_v002"
                ).num_input_examples("validation"),
            ),
        },
    ),
    # RTE
    "rte": Dataset(
        name="rte",
        splits={
            "train": Split(
                name="train",
                size=t5.data.get_mixture_or_task(
                    "glue_rte_v002"
                ).num_input_examples("train"),
            ),
            "validation": Split(
                name="validation",
                size=t5.data.get_mixture_or_task(
                    "glue_rte_v002"
                ).num_input_examples("validation"),
            ),
        },
    ),
    # WNLI
    "wnli": Dataset(
        name="wnli",
        splits={
            "train": Split(
                name="train",
                size=t5.data.get_mixture_or_task(
                    "glue_wnli_v002"
                ).num_input_examples("train"),
            ),
            "validation": Split(
                name="validation",
                size=t5.data.get_mixture_or_task(
                    "glue_wnli_v002"
                ).num_input_examples("validation"),
            ),
        },
    ),
}
"""GLUE datasets."""


SUPER_GLUE_DATASETS = {
    # BoolQ
    "boolq": Dataset(
        name="boolq",
        splits={
            "train": Split(
                name="train",
                size=t5.data.get_mixture_or_task(
                    "super_glue_boolq_v102"
                ).num_input_examples("train"),
            ),
            "validation": Split(
                name="validation",
                size=t5.data.get_mixture_or_task(
                    "super_glue_boolq_v102"
                ).num_input_examples("validation"),
            ),
        },
    ),
    # CommitmentBank
    "cb": Dataset(
        name="cb",
        splits={
            "train": Split(
                name="train",
                size=t5.data.get_mixture_or_task(
                    "super_glue_cb_v102"
                ).num_input_examples("train"),
            ),
            "validation": Split(
                name="validation",
                size=t5.data.get_mixture_or_task(
                    "super_glue_cb_v102"
                ).num_input_examples("validation"),
            ),
        },
    ),
    # COPA
    "copa": Dataset(
        name="copa",
        splits={
            "train": Split(
                name="train",
                size=t5.data.get_mixture_or_task(
                    "super_glue_copa_v102"
                ).num_input_examples("train"),
            ),
            "validation": Split(
                name="validation",
                size=t5.data.get_mixture_or_task(
                    "super_glue_copa_v102"
                ).num_input_examples("validation"),
            ),
        },
    ),
    # MultiRC
    "multirc": Dataset(
        name="multirc",
        splits={
            "train": Split(
                name="train",
                size=t5.data.get_mixture_or_task(
                    "super_glue_multirc_v102"
                ).num_input_examples("train"),
            ),
            "validation": Split(
                name="validation",
                size=t5.data.get_mixture_or_task(
                    "super_glue_multirc_v102"
                ).num_input_examples("validation"),
            ),
        },
    ),
    # ReCoRD
    "record": Dataset(
        name="record",
        splits={
            "train": Split(
                name="train",
                size=t5.data.get_mixture_or_task(
                    "super_glue_record_v102"
                ).num_input_examples("train"),
            ),
            "validation": Split(
                name="validation",
                size=t5.data.get_mixture_or_task(
                    "super_glue_record_v102"
                ).num_input_examples("validation"),
            ),
        },
    ),
    # RTE
    "rte": Dataset(
        name="rte",
        splits={
            "train": Split(
                name="train",
                size=t5.data.get_mixture_or_task(
                    "super_glue_rte_v102"
                ).num_input_examples("train"),
            ),
            "validation": Split(
                name="validation",
                size=t5.data.get_mixture_or_task(
                    "super_glue_rte_v102"
                ).num_input_examples("validation"),
            ),
        },
    ),
    # WiC
    "wic": Dataset(
        name="wic",
        splits={
            "train": Split(
                name="train",
                size=t5.data.get_mixture_or_task(
                    "super_glue_wic_v102"
                ).num_input_examples("train"),
            ),
            "validation": Split(
                name="validation",
                size=t5.data.get_mixture_or_task(
                    "super_glue_wic_v102"
                ).num_input_examples("validation"),
            ),
        },
    ),
}
"""Super GLUE datasets."""


COMMONSENSE_DATASETS = {
    # CommonsenseQA
    "commonsenseqa": Dataset(
        name="commonsenseqa",
        splits={
            "train": Split(name="train", size=9471),
            "validation": Split(name="validation", size=1221),
        },
    ),
    # JHU Ordinal Commonsense Inference
    "joci": Dataset(
        name="joci",
        splits={
            "train": Split(name="train", size=34092),
            "validation": Split(name="validation", size=2500),
        },
    ),
    # ReCoRD
    "record": Dataset(
        name="record",
        splits={
            "train": Split(
                name="train",
                size=t5.data.get_mixture_or_task(
                    "super_glue_record_v102"
                ).num_input_examples("train"),
            ),
            "validation": Split(
                name="validation",
                size=t5.data.get_mixture_or_task(
                    "super_glue_record_v102"
                ).num_input_examples("validation"),
            ),
        },
    ),
}
"""Commonsense datasets besides rainbow."""
