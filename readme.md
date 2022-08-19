Unicorn on Rainbow
==================
Neural models of common sense.

[This repository][source] is for the paper: *Unicorn on Rainbow: A Universal
Commonsense Reasoning Model on a New Multitask Benchmark*. *Unicorn on Rainbow*
introduces a new evaluation, *the cost equivalent curve*, which compares models
in terms of their cost-benefit trade offs. Using cost equivalent curves, we
conduct a large-scale empirical study of intermediate-task transfer for common
sense on a new benchmark collection of commonsense reasoning datasets,
*Rainbow*. With findings from this study, we create a new state-of-the-art
model for commonsense reasoning: *Unicorn*.

Jump to a section of the readme to accomplish different goals:

  - [Rainbow](#rainbow): Read about and download data for *Rainbow*, our new
    commonsense reasoning benchmark.
  - [Unicorn](#unicorn): Get up and running with *Unicorn*, our
    state-of-the-art commonsense reasoning model.
  - [Cost Equivalent Curves](#cost-equivalent-curves): Learn how to generate
    *cost equivalent curves* for your own predictions.
  - [Experimental Results](#experimental-results): Download and analyze the
    results from our hundreds of experiments.
  - [Setup](#setup): Get set up to run the code in this repository.
  - [Quickstart](#quickstart): Run the code in this repo.
  - [Citation](#citation): Cite the *Unicorn on Rainbow* paper.
  - [Contact](#contact): Reach out with questions or comments.

**Note: This repository is intended for research.** There is no intention for
ongoing maintenance.


Rainbow
-------
[*Rainbow*][rainbow-leaderboard] brings together six pre-existing commonsense
reasoning benchmarks: [aNLI][anli-leaderboard],
[Cosmos QA][cosmosqa-leaderboard], [HellaSWAG][hellaswag-leaderboard],
[Physical IQa][physicaliqa-leaderboard], [Social IQa][socialiqa-leaderboard], and
[WinoGrande][winogrande-leaderboard]. These commonsense reasoning benchmarks
span both social and physical common sense.

**Note:** Rainbow pins these datasets to specific versions. To make sure you're
using the correct data, please download those versions below.

### Getting the Data

Rainbow preprocesses all of the datasets into a text-to-text format for ease of
modeling.

  - **Rainbow**: [train/dev/test][rainbow-download] ([checksum][rainbow-checksum])

Alternatively, you can download the individual tasks and preprocess them
yourself.

  - **aNLI**: [train/dev][anli-train-dev-download]
    ([checksum][anli-train-dev-checksum]), [test][anli-test-download]
    ([checksum][anli-test-checksum])
  - **Cosmos QA**: [train/dev/test][cosmosqa-download] ([checksum][cosmosqa-checksum])
  - **HellaSWAG**: [train/dev][hellaswag-train-dev-download]
    ([checksum][hellaswag-train-dev-checksum]), [test][hellaswag-test-download]
    ([checksum][hellaswag-test-checksum])
  - **Physical IQa**: [train/dev][physicaliqa-train-dev-download]
    ([checksum][physicaliqa-train-dev-checksum]),
    [test][physicaliqa-test-download] ([checksum][physicaliqa-test-checksum])
  - **Social IQa**: [train/dev][socialiqa-train-dev-download]
    ([checksum][socialiqa-train-dev-checksum]), [test][socialiqa-test-download]
    ([checksum][socialiqa-test-checksum])
  - **WinoGrande**: [train/dev/test][winogrande-download]
    ([checksum][winogrande-checksum])

All checksums are `sha256`. To compute the checksum with `openssl`, run:

    $ openssl sha256 $FILE_PATH



### Submitting to the Leaderboard

If you develop a model for Rainbow, please feel free to submit to the
[leaderboard][rainbow-leaderboard]!


Unicorn
-------
*Unicorn* (a UNIversal COmmonsense Reasoning Model) solves commonsense
reasoning tasks in the text-to-text format. In principle, Unicorn may be
trained on any NLP task, simply feed it text input and ask it to predict text
output. Unicorn derives from [T5][t5], supercharging it for commonsense
reasoning tasks and achieving state-of-the-art across a number of popular
benchmarks, including [Rainbow](#rainbow) and
[CommonsenseQA][commonsenseqa-leaderboard].

To try Unicorn on your own data, first
[download the weights](#downloading-the-weights) then
[fine-tune and evaluate](#quickstart) it on your own data.

### Downloading the Weights

To run Unicorn, you'll first need to download its weight files into a directory
or path on Google Cloud. Using [`gsutil`][gsutil]:

    gsutil cp -r \
      gs://ai2-mosaic-public/projects/rainbow/v1.0/unicorns/lr-2e-3_batch-size-32
      $DST

Where `$DST` is the destination directory.

### Reproducing our Results

In *Unicorn on Rainbow*, we trained different Unicorns that were first
multitasked on Rainbow using different hyper-parameters. The checkpoint we've
made available had the best performance most often. If you need the other
checkpoints, please email the authors.


Cost Equivalent Curves
----------------------
*Cost equivalent curves* compare the cost-benefit trade offs different
techniques offer. In particular, cost equivalent curves plot the baseline and
new technique's *equivalent costs*, or the costs where they achieve the same
performance. For example, if the cost is measured as the number of examples and
performance is measured by accuracy, then the cost equivalent curve shows how
many examples the baseline needs to match the new technique's accuracy.

The `plot_cost_equivalent_curves` function in
[`bin/create-multi-experiment-figures.py`](./bin/create-multi-experiment-figures.py)
offers example code for how to create cost equivalent curves in Python.


Experimental Results
--------------------
For *Unicorn on Rainbow*, we ran hundreds of experiments. We've made available
the results from all those experiments in order to facilitate future
research. For example, you may want those thousands of training curves to study
hyper-parameter tuning or how loss evolves over training.

Among other things, you'll find:

  - predictions on dev from every checkpoint saved during training
  - training curves (training step vs. loss)
  - learning curves (dataset size vs. accuracy)
  - hyper-parameter tuning
  - all tables and figures from the paper
  - and more...

Our hope is that researchers can reuse this large collection of experiments to
derive new practical and research insights.

### Downloading the Results

Five collections of results are available:

  - **[rainbow-predictions.tar.gz][rainbow-predictions.tar.gz]**
    ([checksum][rainbow-predictions.tar.gz.checksum]): The raw predictions from
    the models we trained.
  - **[rainbow-experiments.tar.gz][rainbow-experiments.tar.gz]**
    ([checksum][rainbow-experiments.tar.gz.checksum]): A better organized collection
    of the raw predictions into the experiments (some predictions are
    duplicated since they're relevant to multiple experiments).
  - **[rainbow-results.tar.gz][rainbow-results.tar.gz]**
    ([checksum][rainbow-results.tar.gz.checksum]): A collection of training
    curves (JSON Lines formatted) and tables of results (CSV formatted)
    providing the results for each of the experiments.
  - **[rainbow-figures.tar.gz][rainbow-figures.tar.gz]**
    ([checksum][rainbow-figures.tar.gz.checksum]): Figures and plots
    visualizing the results, including both the figures from the paper and
    many additional figures.
  - **[rainbow-latex-tables.tar.gz][rainbow-latex-tables.tar.gz]**
    ([checksum][rainbow-latex-tables.tar.gz.checksum]): Automatically generated
    latex tables for the results.

All checksums are `sha256`. To compute the checksum with `openssl`, run:

    $ openssl sha256 $FILE_PATH

**NOTE:** The learning curves experiments varied the number of training
examples up to 16,000; however, CommonsenseQA has fewer than 16,000 training
examples. Thus, for CommonsenseQA numbers higher than 9,741 are truncated to
that size. This subtlety is taken care of by the data processing pipeline when
the experiments are processed into the results tables, so it only affects
`rainbow-predictions.tar.gz` and `rainbow-experiments.tar.gz`.


### Replicating Our Analysis Pipeline

All the scripts to replicate our analysis pipeline reside in
[bin/](./bin/). In order to run the scripts, you'll need to get
[set up](#setup) for development.

The overall pipeline is as follows:

    +----------------------------+
    | rainbow-predictions.tar.gz |
    +----------------------------+
                  |
                  | (bin/organize-experiments)
                  V
    +----------------------------+
    | rainbow-experiments.tar.gz |
    +----------------------------+
                  |
                  | (bin/generate-tables.py)
                  V
      +------------------------+
      | rainbow-results.tar.gz |
      +------------------------+
             |         |
             |         | (bin/generate-latex-tables.py)
             |         V
             |     +-----------------------------+
             |     | rainbow-latex-tables.tar.gz |
             |     +-----------------------------+
             |
             | (bin/create-single-experiment-figures.py)
             | (bin/create-multi-experiment-figures.py)
             V
    +------------------------+
    | rainbow-figures.tar.gz |
    +------------------------+

To run the pipeline, start by downloading `rainbow-predictions.tar.gz` (see
[Downloading the Results](#downloading-the-results) above).

Use `bin/organize-experiments` to produce `rainbow-experiments.tar.gz`:

    $ tar -xf rainbow-predictions.tar.gz
    $ bin/organize-experiments rainbow-predictions $DST

Where `$DST` is the desired destination directory (for example the current
directory, `.`).

Use `bin/generate-tables.py` to produce `rainbow-results.tar.gz`:

    $ bin/generate-tables.py rainbow-experiments rainbow-results

Use `bin/create-single-experiment-figures.py` and
`bin/create-multi-experiment-figures.py` to create `rainbow-figures.tar.gz`:

    $ bin/create-single-experiment-figures.py rainbow-results rainbow-figures/single-experiment
    $ bin/create-multi-experiment-figures.py rainbow-results rainbow-figures/multi-experiment

And use `bin/generate-latex-tables.py` to produce
`rainbow-latex-tables.tar.gz`:

    $ bin/generate-latex-tables.py rainbow-results rainbow-latex-tables

All scripts except `bin/organize-experiments` are also self-documenting, so
pass `--help` to any of them for more information.


Setup
-----
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



Quickstart
----------
Before following this section, make sure you've done the [Setup](#setup).

### Fine-tuning

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

### Evaluation

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



### Tests and Code Quality

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



Citation
--------
*Unicorn on Rainbow* was published at AAAI-2021. Please use the following bibtex entry to refer to the paper:
```
@article{Lourie2021UNICORNOR,
  title={UNICORN on RAINBOW: A Universal Commonsense Reasoning Model on a New Multitask Benchmark},
  author={Nicholas Lourie and Ronan {Le Bras} and Chandra Bhagavatula and Yejin Choi},
  journal={AAAI},
  year={2021}
}
```


Contact
-------
For public, non-sensitive questions and concerns, please file an issue on this
repository.

For private or sensitive inquiries email mosaic on the allenai.org website.


[anli-leaderboard]: https://leaderboard.allenai.org/anli
[anli-test-checksum]: https://storage.googleapis.com/ai2-mosaic-public/projects/rainbow/v1.0/data/raw/anli/alphanli-test.zip.checksum
[anli-test-download]: https://storage.googleapis.com/ai2-mosaic-public/projects/rainbow/v1.0/data/raw/anli/alphanli-test.zip
[anli-train-dev-checksum]: https://storage.googleapis.com/ai2-mosaic-public/projects/rainbow/v1.0/data/raw/anli/alphanli-train-dev.zip.checksum
[anli-train-dev-download]: https://storage.googleapis.com/ai2-mosaic-public/projects/rainbow/v1.0/data/raw/anli/alphanli-train-dev.zip
[bin/evaluate.py]: ./bin/evaluate.py
[bin/fine-tune.py]: ./bin/fine-tune.py
[bin/format]: ./bin/format
[bin/verify]: ./bin/verify
[black]: https://black.readthedocs.io/en/stable/
[commonsenseqa-leaderboard]: https://www.tau-nlp.org/csqa-leaderboard
[cosmosqa-checksum]: https://storage.googleapis.com/ai2-mosaic-public/projects/rainbow/v1.0/data/raw/cosmosqa/cosmosqa-data.zip.checksum
[cosmosqa-download]: https://storage.googleapis.com/ai2-mosaic-public/projects/rainbow/v1.0/data/raw/cosmosqa/cosmosqa-data.zip
[cosmosqa-leaderboard]: https://leaderboard.allenai.org/cosmosqa
[gsutil]: https://cloud.google.com/storage/docs/quickstart-gsutil
[hellaswag-leaderboard]: https://leaderboard.allenai.org/hellaswag
[hellaswag-test-checksum]: https://storage.googleapis.com/ai2-mosaic-public/projects/rainbow/v1.0/data/raw/hellaswag/hellaswag-test.zip.checksum
[hellaswag-test-download]: https://storage.googleapis.com/ai2-mosaic-public/projects/rainbow/v1.0/data/raw/hellaswag/hellaswag-test.zip
[hellaswag-train-dev-checksum]: https://storage.googleapis.com/ai2-mosaic-public/projects/rainbow/v1.0/data/raw/hellaswag/hellaswag-train-dev.zip.checksum
[hellaswag-train-dev-download]: https://storage.googleapis.com/ai2-mosaic-public/projects/rainbow/v1.0/data/raw/hellaswag/hellaswag-train-dev.zip
[physicaliqa-leaderboard]: https://leaderboard.allenai.org/physicaliqa
[physicaliqa-test-checksum]: https://storage.googleapis.com/ai2-mosaic-public/projects/rainbow/v1.0/data/raw/physicaliqa/physicaliqa-test.zip.checksum
[physicaliqa-test-download]: https://storage.googleapis.com/ai2-mosaic-public/projects/rainbow/v1.0/data/raw/physicaliqa/physicaliqa-test.zip
[physicaliqa-train-dev-checksum]: https://storage.googleapis.com/ai2-mosaic-public/projects/rainbow/v1.0/data/raw/physicaliqa/physicaliqa-train-dev.zip.checksum
[physicaliqa-train-dev-download]: https://storage.googleapis.com/ai2-mosaic-public/projects/rainbow/v1.0/data/raw/physicaliqa/physicaliqa-train-dev.zip
[pytest]: https://docs.pytest.org/en/latest/
[rainbow-checksum]: https://storage.googleapis.com/ai2-mosaic-public/projects/rainbow/v1.0/data/text-to-text/v1.0.rainbow.tar.gz.checksum
[rainbow-download]: https://storage.googleapis.com/ai2-mosaic-public/projects/rainbow/v1.0/data/text-to-text/v1.0.rainbow.tar.gz
[rainbow-experiments.tar.gz.checksum]: https://storage.googleapis.com/ai2-mosaic-public/projects/rainbow/v1.0/experiments/rainbow-experiments.tar.gz.checksum
[rainbow-experiments.tar.gz]: https://storage.googleapis.com/ai2-mosaic-public/projects/rainbow/v1.0/experiments/rainbow-experiments.tar.gz
[rainbow-figures.tar.gz.checksum]: https://storage.googleapis.com/ai2-mosaic-public/projects/rainbow/v1.0/experiments/rainbow-figures.tar.gz.checksum
[rainbow-figures.tar.gz]: https://storage.googleapis.com/ai2-mosaic-public/projects/rainbow/v1.0/experiments/rainbow-figures.tar.gz
[rainbow-latex-tables.tar.gz.checksum]: https://storage.googleapis.com/ai2-mosaic-public/projects/rainbow/v1.0/experiments/rainbow-latex-tables.tar.gz.checksum
[rainbow-latex-tables.tar.gz]: https://storage.googleapis.com/ai2-mosaic-public/projects/rainbow/v1.0/experiments/rainbow-latex-tables.tar.gz
[rainbow-leaderboard]: https://leaderboard.allenai.org/rainbow
[rainbow-predictions.tar.gz.checksum]: https://storage.googleapis.com/ai2-mosaic-public/projects/rainbow/v1.0/experiments/rainbow-predictions.tar.gz.checksum
[rainbow-predictions.tar.gz]: https://storage.googleapis.com/ai2-mosaic-public/projects/rainbow/v1.0/experiments/rainbow-predictions.tar.gz
[rainbow-results.tar.gz.checksum]: https://storage.googleapis.com/ai2-mosaic-public/projects/rainbow/v1.0/experiments/rainbow-results.tar.gz.checksum
[rainbow-results.tar.gz]: https://storage.googleapis.com/ai2-mosaic-public/projects/rainbow/v1.0/experiments/rainbow-results.tar.gz
[socialiqa-leaderboard]: https://leaderboard.allenai.org/socialiqa
[socialiqa-test-checksum]: https://storage.googleapis.com/ai2-mosaic-public/projects/rainbow/v1.0/data/raw/socialiqa/socialiqa-test.zip.checksum
[socialiqa-test-download]: https://storage.googleapis.com/ai2-mosaic-public/projects/rainbow/v1.0/data/raw/socialiqa/socialiqa-test.zip
[socialiqa-train-dev-checksum]: https://storage.googleapis.com/ai2-mosaic-public/projects/rainbow/v1.0/data/raw/socialiqa/socialiqa-train-dev.zip.checksum
[socialiqa-train-dev-download]: https://storage.googleapis.com/ai2-mosaic-public/projects/rainbow/v1.0/data/raw/socialiqa/socialiqa-train-dev.zip
[source]: https://github.com/allenai/rainbow
[t5]: https://github.com/google-research/text-to-text-transfer-transformer
[winogrande-checksum]: https://storage.googleapis.com/ai2-mosaic-public/projects/rainbow/v1.0/data/raw/winogrande/winogrande_1.1.zip.checksum
[winogrande-download]: https://storage.googleapis.com/ai2-mosaic-public/projects/rainbow/v1.0/data/raw/winogrande/winogrande_1.1.zip
[winogrande-leaderboard]: https://leaderboard.allenai.org/winogrande
