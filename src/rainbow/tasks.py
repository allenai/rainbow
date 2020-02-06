"""Task definitions for rainbow."""

import os

import t5
import tensorflow as tf

from . import core, datasets, preprocessors, settings


# socialiqa

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
