"""Task definitions for rainbow."""

import os

import t5
import tensorflow as tf

from . import core, datasets, preprocessors, settings


# Create tasks for the datasets.

for dataset in datasets.RAINBOW_DATASETS.values():
    for size in settings.LEARNING_CURVE_SIZES:
        task_name = (
            f"{dataset.name}_task"
            if size is None
            else f"{dataset.name}_{size}_task"
        )
        t5.data.TaskRegistry.add(
            name=task_name,
            task_cls=core.CsvTask,
            # args for CsvTask
            #   dataset configuration and location
            split_to_filepattern={
                split.name: os.path.join(
                    settings.PREPROCESSED_DATASETS_DIR,
                    dataset.name,
                    settings.PREPROCESSED_SPLIT_FILE_NAME_TEMPLATE.format(
                        split=split.name, dataset=dataset.name
                    ),
                )
                for split in dataset.splits.values()
            },
            num_input_examples={
                split.name: split.size for split in dataset.splits.values()
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
            #   dataset truncation
            truncate_to=size,
            # args for the task class
            postprocess_fn=t5.data.postprocessors.lower_text,
        )
