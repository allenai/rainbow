#! /usr/bin/env python

"""Evaluate the model on the rainbow datasets."""

import logging

import click
import t5
import tensorflow as tf

from rainbow import utils

import rainbow.mixtures

# N.B. We must import rainbow.mixtures here so that the mixtures are registered
# and available for evaluation.


logger = logging.getLogger(__name__)


@click.command()
@click.argument("mixture", type=str)
@click.argument("results_dir", type=str)
@click.option(
    "--split",
    type=str,
    default="validation",
    help="The split on which to evaluate. Defaults to 'validation'.",
)
@click.option(
    "--checkpoint-steps",
    type=str,
    default="all",
    help="The checkpoint steps on which to evaluate. Defaults to 'all'.",
)
@click.option(
    "--batch-size",
    type=int,
    default=64,
    help=(
        "The batch size to use for prediction. For efficient prediction on the"
        " TPU, choose a multiple of either 8 or 128. Defaults to 64."
    ),
)
@click.option(
    "--model-parallelism",
    type=int,
    default=8,
    help="The degree of model parallelism to use. Defaults to 8.",
)
@click.option(
    "--tpu-name",
    type=str,
    required=True,
    envvar="TPU_NAME",
    help=(
        "The name of the TPU. Defaults to the TPU_NAME environment variable."
    ),
)
@click.option(
    "--tpu-topology",
    type=str,
    required=True,
    envvar="TPU_TOPOLOGY",
    help=(
        "The topology of the TPU. Defaults to the TPU_TOPOLOGY environment"
        " variable."
    ),
)
def evaluate(
    mixture: str,
    results_dir: str,
    split: str,
    checkpoint_steps: str,
    batch_size: int,
    model_parallelism: int,
    tpu_name: str,
    tpu_topology: str,
) -> None:
    """Evaluate the model located at RESULTS_DIR on MIXTURE."""
    utils.configure_logging(clear=True)

    # Validate arguments.

    if not results_dir.startswith("gs://"):
        raise ValueError(f"RESULTS_DIR ({results_dir}) must be a GCS path.")
    elif not tf.io.gfile.exists(results_dir):
        raise IOError(f"RESULTS_DIR ({results_dir}) doesn't exist.")

    # Run evaluation.

    model = t5.models.MtfModel(
        model_dir=results_dir,
        tpu=tpu_name,
        tpu_topology=tpu_topology,
        model_parallelism=model_parallelism,
        batch_size=batch_size,
        sequence_length={"inputs": 512, "targets": 512},
        learning_rate_schedule=None,
        save_checkpoints_steps=5000,
        keep_checkpoint_max=None,
        iterations_per_loop=100,
    )

    model.eval(
        mixture_or_task_name=mixture,
        checkpoint_steps=checkpoint_steps,
        split=split,
    )


if __name__ == "__main__":
    evaluate()  # pylint: disable=no-value-for-parameter
