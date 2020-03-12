"""Core classes and functions used throughout rainbow."""

from typing import Callable, Dict, List, Optional, Sequence, Union

import t5
import tensorflow as tf

from . import utils


class CsvTask(t5.data.Task):
    """A ``Task`` for CSV formatted datasets.

    Parameters
    ----------
    name : str, required
        The name of the task. It must be unique.
    split_to_filepattern : Dict[str, str], required
        A dictionary mapping each split (``"train"``, ``"validation"``, and
        ``"test"``) to a file pattern (glob) matching all the files for that
        split.
    num_input_examples : Dict[str, int], required
        A dictionary mapping each split's name to the number of input
        examples in the split.
    text_preprocessor : Union[Callable, Sequence[Callable]], required
        The text preprocessor function or a sequence of such functions.
    sentencepiece_model_path : str, required
        The path to the sentence piece model.
    metric_fns : Sequence[Callable], required
        A sequence of metric functions for the task.
    record_defaults : List[Union[tf.DType, tf.Tensor]], required
        See ``tf.data.experimental.CsvDataset``.
    compression_type : Optional[str], optional (default=None)
        See ``tf.data.experimental.CsvDataset``. Defaults to ``None``.
    buffer_size : Optional[int], optional (default=None)
        See ``tf.data.experimental.CsvDataset``. Defaults to ``None``.
    header : bool, optional (default=False)
        See ``tf.data.experimental.CsvDataset``. Defaults to ``False``.
    field_delim : str, optional (default=",")
        See ``tf.data.experimental.CsvDataset``. Defaults to ",".
    use_quote_delim : bool, optional (default=True)
        See ``tf.data.experimental.CsvDataset``. Defaults to ``True``.
    na_value : str, optional (default="")
        See ``tf.data.experimental.CsvDataset``. Defaults to ``""``.
    select_cols : Optional[List[int]], optional (default=None)
        See ``tf.data.experimental.CsvDataset``. Defaults to ``None``.
    truncate_to : Optional[int], optional (default=None)
        The number of examples the training set should have after
        truncation. To use truncation, the training set's split MUST be
        called "train". Truncation samples a random subset of the data
        in a deterministic way using a PRNG, based on the name of the
        dataset, it's original size, and the desired truncated size. If
        ``truncate_to`` is ``None``, the full training set will be used.
    **kwargs
        Additional keyword arguments passed to the super class's (``Task``)
        constructor.
    """

    def __init__(
        self,
        name: str,
        split_to_filepattern: Dict[str, str],
        num_input_examples: Dict[str, int],
        text_preprocessor: Union[Callable, Sequence[Callable]],
        sentencepiece_model_path: str,
        metric_fns: Sequence[Callable],
        record_defaults: List[Union[tf.DType, tf.Tensor]],
        compression_type: Optional[str] = None,
        buffer_size: Optional[int] = None,
        header: bool = False,
        field_delim: str = ",",
        use_quote_delim: bool = True,
        na_value: str = "",
        select_cols: Optional[List[int]] = None,
        truncate_to: Optional[int] = None,
        **kwargs,
    ) -> None:
        def dataset_fn(split, shuffle_files=False):
            """A function for creating the Datset for the split."""
            # N.B. The shuffle_files argument is ignored, and only used
            # to make the function compatible with the required API. The
            # file names cannot be shuffled because then the truncate_to
            # argument would not subsample the dataset
            # deterministically.

            # Define a function for reading the csv parts.
            def _read_part(fname):
                return tf.data.experimental.CsvDataset(
                    filenames=fname,
                    record_defaults=record_defaults,
                    compression_type=compression_type,
                    buffer_size=buffer_size,
                    header=header,
                    field_delim=field_delim,
                    use_quote_delim=use_quote_delim,
                    na_value=na_value,
                    select_cols=select_cols,
                )

            # Construct the file paths for the CSV parts.
            split_fpaths = tf.data.Dataset.list_files(
                file_pattern=split_to_filepattern[split], shuffle=False,
            )

            # Construct the full split by flat_mapping the dataset parts. Use
            # flat_map so that we preserve the order, i.e. the resulting
            # dataset is the same as you'd get by concatenating the CSVs.
            dataset = split_fpaths.flat_map(_read_part)

            # Optionally truncate the dataset.
            if truncate_to is not None and split == "train":
                dataset = dataset.shuffle(
                    buffer_size=num_input_examples[split],
                    seed=utils.string_to_seed(f"{name}_{truncate_to}"),
                    reshuffle_each_iteration=False,
                ).take(count=truncate_to)

            return dataset

        super().__init__(
            name=name,
            dataset_fn=dataset_fn,
            splits=split_to_filepattern.keys(),
            text_preprocessor=text_preprocessor,
            sentencepiece_model_path=sentencepiece_model_path,
            metric_fns=metric_fns,
            num_input_examples=num_input_examples,
            **kwargs,
        )
