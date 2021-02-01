# Analyzing Results

Documentation for analyzing the results from `rainbow`.

## What's Available

We've made available all results from all of our experiments. These include
hyper-parameter optimization, predictions from every checkpoint saved during
training, and other more organized and processed forms of the results. Our hope
is that other people can reuse this large collection of experiments to derive
new practical and research insights.

## Downloading the Data

We've made available five collections of results for download:

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

[rainbow-predictions.tar.gz]: https://storage.googleapis.com/ai2-mosaic-public/projects/rainbow/v1.0/experiments/rainbow-predictions.tar.gz
[rainbow-predictions.tar.gz.checksum]: https://storage.googleapis.com/ai2-mosaic-public/projects/rainbow/v1.0/experiments/rainbow-predictions.tar.gz.checksum
[rainbow-experiments.tar.gz]: https://storage.googleapis.com/ai2-mosaic-public/projects/rainbow/v1.0/experiments/rainbow-experiments.tar.gz
[rainbow-experiments.tar.gz.checksum]: https://storage.googleapis.com/ai2-mosaic-public/projects/rainbow/v1.0/experiments/rainbow-experiments.tar.gz.checksum
[rainbow-results.tar.gz]: https://storage.googleapis.com/ai2-mosaic-public/projects/rainbow/v1.0/experiments/rainbow-results.tar.gz
[rainbow-results.tar.gz.checksum]: https://storage.googleapis.com/ai2-mosaic-public/projects/rainbow/v1.0/experiments/rainbow-results.tar.gz.checksum
[rainbow-figures.tar.gz]: https://storage.googleapis.com/ai2-mosaic-public/projects/rainbow/v1.0/experiments/rainbow-figures.tar.gz
[rainbow-figures.tar.gz.checksum]: https://storage.googleapis.com/ai2-mosaic-public/projects/rainbow/v1.0/experiments/rainbow-figures.tar.gz.checksum
[rainbow-latex-tables.tar.gz]: https://storage.googleapis.com/ai2-mosaic-public/projects/rainbow/v1.0/experiments/rainbow-latex-tables.tar.gz
[rainbow-latex-tables.tar.gz.checksum]: https://storage.googleapis.com/ai2-mosaic-public/projects/rainbow/v1.0/experiments/rainbow-latex-tables.tar.gz.checksum

## Replicating Our Analysis Pipeline

All the scripts to replicate our analysis pipeline reside in
[bin/](../bin/). In order to run the scripts, you'll need to get set up for
[development](./development.md).

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
[Downloading the Data](#downloading-the-data) above).

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
