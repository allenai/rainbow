# Development

Development documentation for `rainbow`.

## Setup

This project requires Python 3.6 or above.

First, install the project's dependencies:

    ./bin/install

Next, make sure you have the following environment variables set:

1. `RAINBOW_DATASETS_DIR`: The directory for storing all relevant datasets.
2. `RAINBOW_PREPROCESSED_DATASETS_DIR`: The directory for storing the
   preprocessed dataset split files.
3. `RAINBOW_TFDS_DATASETS_DIR`: The directory for storing the TFDS
   (tensorflow datasets) datasets.

Training requires TPUs. For training, all directories should point to Google
Cloud Storage prefixes. Additionally, you'll need the following environment
variables:

1. `PROJECT`: Your Google Cloud project's ID.
2. `ZONE`: Your Google Cloud virtual machine's zone.
3. `TPU_NAME`: Your TPU's name.
4. `TPU_TOPOLOGY`: Your TPU's topology.

Then, download and prepare all the datasets for text-to-text modeling:

    $ ./bin/prepare.py --help
    Usage: prepare.py [OPTIONS]

      Prepare all relevant datasets for text-to-text modeling.

      Download to and read the datasets from --src, transform them into CSVs
      suitable for text-to-text models, then write the results to --dst. Google
      storage paths are supported.

    Options:
      --src TEXT        The directory to which to download all the relevant
                        datasets. Defaults to the RAINBOW_DATASETS_DIR environment
                        variable.  [required]
      --dst TEXT        The directory to which to write the preprocessed dataset
                        files. Defaults to the RAINBOW_PREPROCESSED_DATASETS_DIR
                        environment variable.  [required]
      --force-download  Force downloads of all the datasets, otherwise only
                        missing datasets will be downloaded.
      --help            Show this message and exit.

Finally, verify your installation:

    ./bin/verify

## Fine-tuning and Evaluation

To fine-tune the model, use [`bin/fine-tune.py`][bin/fine-tune.py]:

    $ ./bin/fine-tune.py --help
    Usage: fine-tune.py [OPTIONS] MIXTURE RESULTS_DIR

      Fine-tune the model on MIXTURE, writing results to RESULTS_DIR.

    Options:
      --pretrained-model TEXT         The path to or name of the pretrained model.
                                      Defaults to 3B.
      --n-steps INTEGER               The number of gradient updates. Defaults to
                                      25,000.
      --learning-rate FLOAT           The learning rate to use for training.
                                      Defaults to 3e-3.
      --batch-size INTEGER            The batch size to use for training. For
                                      efficient training on the TPU, choose a
                                      multiple of either 8 or 128. Defaults to 16.
      --model-parallelism INTEGER     The degree of model parallelism to use.
                                      Defaults to 8.
      --save-checkpoints-steps INTEGER
                                      The number of steps to take before saving a
                                      checkpoint. Defaults to 5000.
      --n-checkpoints-to-keep INTEGER
                                      The number of checkpoints to keep during
                                      fine-tuning. Defaults to 4.
      --tpu-name TEXT                 The name of the TPU. Defaults to the
                                      TPU_NAME environment variable.  [required]
      --tpu-topology TEXT             The topology of the TPU. Defaults to the
                                      TPU_TOPOLOGY environment variable.
                                      [required]
      --help                          Show this message and exit.

To evaluate the model, use [`bin/evaluate.py`][bin/evaluate.py]:

    $ ./bin/evaluate.py --help
    Usage: evaluate.py [OPTIONS] MIXTURE RESULTS_DIR

      Evaluate the model located at RESULTS_DIR on MIXTURE.

    Options:
      --batch-size INTEGER         The batch size to use for prediction. For
                                   efficient prediction on the TPU, choose a
                                   multiple of either 8 or 128. Defaults to 64.
      --model-parallelism INTEGER  The degree of model parallelism to use.
                                   Defaults to 8.
      --tpu-name TEXT              The name of the TPU. Defaults to the TPU_NAME
                                   environment variable.  [required]
      --tpu-topology TEXT          The topology of the TPU. Defaults to the
                                   TPU_TOPOLOGY environment variable.  [required]
      --help                       Show this message and exit.

[bin/fine-tune.py]: ../bin/fine-tune.py
[bin/evaluate.py]: ../bin/evaluate.py

## Tests and Code Quality

The code is formatted with [black][black]. You can run the formatter using the
[`bin/format`][bin/format] script:

    $ ./bin/format

To run code quality checks, use the [`bin/verify`][bin/verify] script:

    $ ./bin/verify

For fine-grained control of which tests to run, use [`pytest`][pytest]
directly:

    $ pytest

You can also skip slower tests by passing the `--skip-slow` (`-s`) flag:

    $ pytest --skip-slow

[black]: https://black.readthedocs.io/en/stable/
[bin/format]: ../bin/format
[bin/verify]: ../bin/verify
[pytest]: https://docs.pytest.org/en/latest/
