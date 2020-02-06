"""Task definitions for rainbow."""

import os

import t5
import tensorflow as tf

from . import core, datasets, preprocessors, settings


# AlphaNLI

_anli = datasets.RAINBOW_DATASETS["anli"]

t5.data.TaskRegistry.add(
    name="anli",
    task_cls=core.CsvTask,
    # args for CsvTask
    #   dataset configuration and location
    split_to_filepattern={
        split.name: os.path.join(
            settings.PREPROCESSED_DATASETS_DIR,
            _anli.name,
            settings.PREPROCESSED_SPLIT_FILE_NAME_TEMPLATE.format(
                split=split.name, dataset=_anli.name
            ),
        )
        for split in _anli.splits.values()
    },
    text_preprocessor=[
        preprocessors.make_add_field_names_preprocessor(
            field_indices=[1, 2], field_names=["inputs", "targets"]
        )
    ],
    sentencepiece_model_path=t5.data.DEFAULT_SPM_PATH,
    metric_fns=[t5.evaluation.metrics.accuracy],
    #   CSV parsing
    record_defaults=[tf.int32, tf.string, tf.string],
    compression_type=None,
    buffer_size=None,
    header=True,
    field_delim=",",
    use_quote_delim=True,
    na_value="",
    select_cols=None,
    # args for the task class
    postprocess_fn=t5.data.postprocessors.lower_text,
    num_input_examples={
        split.name: split.size for split in _anli.splits.values()
    },
)
"""The AlphaNLI Task."""


# CosmosQA

_cosmosqa = datasets.RAINBOW_DATASETS["cosmosqa"]

t5.data.TaskRegistry.add(
    name="cosmosqa",
    task_cls=core.CsvTask,
    # args for CsvTask
    #   dataset configuration and location
    split_to_filepattern={
        split.name: os.path.join(
            settings.PREPROCESSED_DATASETS_DIR,
            _cosmosqa.name,
            settings.PREPROCESSED_SPLIT_FILE_NAME_TEMPLATE.format(
                split=split.name, dataset=_cosmosqa.name
            ),
        )
        for split in _cosmosqa.splits.values()
    },
    text_preprocessor=[
        preprocessors.make_add_field_names_preprocessor(
            field_indices=[1, 2], field_names=["inputs", "targets"]
        )
    ],
    sentencepiece_model_path=t5.data.DEFAULT_SPM_PATH,
    metric_fns=[t5.evaluation.metrics.accuracy],
    #   CSV parsing
    record_defaults=[tf.int32, tf.string, tf.string],
    compression_type=None,
    buffer_size=None,
    header=True,
    field_delim=",",
    use_quote_delim=True,
    na_value="",
    select_cols=None,
    # args for the task class
    postprocess_fn=t5.data.postprocessors.lower_text,
    num_input_examples={
        split.name: split.size for split in _cosmosqa.splits.values()
    },
)
"""The CosmosQA Task."""


# HellaSWAG

_hellaswag = datasets.RAINBOW_DATASETS["hellaswag"]

t5.data.TaskRegistry.add(
    name="hellaswag",
    task_cls=core.CsvTask,
    # args for CsvTask
    #   dataset configuration and location
    split_to_filepattern={
        split.name: os.path.join(
            settings.PREPROCESSED_DATASETS_DIR,
            _hellaswag.name,
            settings.PREPROCESSED_SPLIT_FILE_NAME_TEMPLATE.format(
                split=split.name, dataset=_hellaswag.name
            ),
        )
        for split in _hellaswag.splits.values()
    },
    text_preprocessor=[
        preprocessors.make_add_field_names_preprocessor(
            field_indices=[1, 2], field_names=["inputs", "targets"]
        )
    ],
    sentencepiece_model_path=t5.data.DEFAULT_SPM_PATH,
    metric_fns=[t5.evaluation.metrics.accuracy],
    #   CSV parsing
    record_defaults=[tf.int32, tf.string, tf.string],
    compression_type=None,
    buffer_size=None,
    header=True,
    field_delim=",",
    use_quote_delim=True,
    na_value="",
    select_cols=None,
    # args for the task class
    postprocess_fn=t5.data.postprocessors.lower_text,
    num_input_examples={
        split.name: split.size for split in _hellaswag.splits.values()
    },
)
"""The HellaSWAG Task."""


# PhysicalIQA

_physicaliqa = datasets.RAINBOW_DATASETS["physicaliqa"]

t5.data.TaskRegistry.add(
    name="physicaliqa",
    task_cls=core.CsvTask,
    # args for CsvTask
    #   dataset configuration and location
    split_to_filepattern={
        split.name: os.path.join(
            settings.PREPROCESSED_DATASETS_DIR,
            _physicaliqa.name,
            settings.PREPROCESSED_SPLIT_FILE_NAME_TEMPLATE.format(
                split=split.name, dataset=_physicaliqa.name
            ),
        )
        for split in _physicaliqa.splits.values()
    },
    text_preprocessor=[
        preprocessors.make_add_field_names_preprocessor(
            field_indices=[1, 2], field_names=["inputs", "targets"]
        )
    ],
    sentencepiece_model_path=t5.data.DEFAULT_SPM_PATH,
    metric_fns=[t5.evaluation.metrics.accuracy],
    #   CSV parsing
    record_defaults=[tf.int32, tf.string, tf.string],
    compression_type=None,
    buffer_size=None,
    header=True,
    field_delim=",",
    use_quote_delim=True,
    na_value="",
    select_cols=None,
    # args for the task class
    postprocess_fn=t5.data.postprocessors.lower_text,
    num_input_examples={
        split.name: split.size for split in _physicaliqa.splits.values()
    },
)
"""The PhysicalIQA Task."""


# SocialIQA

_socialiqa = datasets.RAINBOW_DATASETS["socialiqa"]

t5.data.TaskRegistry.add(
    name="socialiqa",
    task_cls=core.CsvTask,
    # args for CsvTask
    #   dataset configuration and location
    split_to_filepattern={
        split.name: os.path.join(
            settings.PREPROCESSED_DATASETS_DIR,
            _socialiqa.name,
            settings.PREPROCESSED_SPLIT_FILE_NAME_TEMPLATE.format(
                split=split.name, dataset=_socialiqa.name
            ),
        )
        for split in _socialiqa.splits.values()
    },
    text_preprocessor=[
        preprocessors.make_add_field_names_preprocessor(
            field_indices=[1, 2], field_names=["inputs", "targets"]
        )
    ],
    sentencepiece_model_path=t5.data.DEFAULT_SPM_PATH,
    metric_fns=[t5.evaluation.metrics.accuracy],
    #   CSV parsing
    record_defaults=[tf.int32, tf.string, tf.string],
    compression_type=None,
    buffer_size=None,
    header=True,
    field_delim=",",
    use_quote_delim=True,
    na_value="",
    select_cols=None,
    # args for the task class
    postprocess_fn=t5.data.postprocessors.lower_text,
    num_input_examples={
        split.name: split.size for split in _socialiqa.splits.values()
    },
)
"""The SocialIQA Task."""


# WinoGrande

_winogrande = datasets.RAINBOW_DATASETS["winogrande"]

t5.data.TaskRegistry.add(
    name="winogrande",
    task_cls=core.CsvTask,
    # args for CsvTask
    #   dataset configuration and location
    split_to_filepattern={
        split.name: os.path.join(
            settings.PREPROCESSED_DATASETS_DIR,
            _winogrande.name,
            settings.PREPROCESSED_SPLIT_FILE_NAME_TEMPLATE.format(
                split=split.name, dataset=_winogrande.name
            ),
        )
        for split in _winogrande.splits.values()
    },
    text_preprocessor=[
        preprocessors.make_add_field_names_preprocessor(
            field_indices=[1, 2], field_names=["inputs", "targets"]
        )
    ],
    sentencepiece_model_path=t5.data.DEFAULT_SPM_PATH,
    metric_fns=[t5.evaluation.metrics.accuracy],
    #   CSV parsing
    record_defaults=[tf.int32, tf.string, tf.string],
    compression_type=None,
    buffer_size=None,
    header=True,
    field_delim=",",
    use_quote_delim=True,
    na_value="",
    select_cols=None,
    # args for the task class
    postprocess_fn=t5.data.postprocessors.lower_text,
    num_input_examples={
        split.name: split.size for split in _winogrande.splits.values()
    },
)
"""The WinoGrande Task."""
