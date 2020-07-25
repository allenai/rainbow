#! /usr/bin/env python

"""Fine-tune the model on the rainbow datasets."""

import logging

import click
import t5
import tensorflow as tf

from rainbow import utils

import rainbow.mixtures

# N.B. We must import rainbow.mixtures here so that the mixtures are registered
# and available for training.


logger = logging.getLogger(__name__)


PRETRAINED_MODELS = {
    "small": "gs://t5-data/pretrained_models/small",
    "base": "gs://t5-data/pretrained_models/base",
    "large": "gs://t5-data/pretrained_models/large",
    "3B": "gs://t5-data/pretrained_models/3B",
    "11B": "gs://t5-data/pretrained_models/11B",
}


@click.command()
@click.argument("mixture", type=str)
@click.argument("results_dir", type=str)
@click.option(
    "--split",
    type=str,
    default="train",
    help="The split on which to train. Defaults to 'train'.",
)
@click.option(
    "--pretrained-model",
    type=str,
    default="3B",
    help="The path to or name of the pretrained model. Defaults to 3B.",
)
@click.option(
    "--n-steps",
    type=int,
    default=25000,
    help="The number of gradient updates. Defaults to 25,000.",
)
@click.option(
    "--learning-rate",
    type=float,
    default=3e-3,
    help="The learning rate to use for training. Defaults to 3e-3.",
)
@click.option(
    "--batch-size",
    type=int,
    default=16,
    help=(
        "The batch size to use for training. For efficient training on the"
        " TPU, choose a multiple of either 8 or 128. Defaults to 16."
    ),
)
@click.option(
    "--model-parallelism",
    type=int,
    default=8,
    help="The degree of model parallelism to use. Defaults to 8.",
)
@click.option(
    "--save-checkpoints-steps",
    type=int,
    default=5000,
    help=(
        "The number of steps to take before saving a checkpoint. Defaults to"
        " 5000."
    ),
)
@click.option(
    "--n-checkpoints-to-keep",
    type=int,
    default=4,
    help=(
        "The number of checkpoints to keep during fine-tuning. Defaults"
        " to 4."
    ),
)
@click.option(
    "--tpu-name",
    type=str,
    required=True,
    envvar="TPU_NAME",
    help="The name of the TPU. Defaults to the TPU_NAME environment variable.",
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
def fine_tune(
    mixture: str,
    results_dir: str,
    split: str,
    pretrained_model: str,
    n_steps: int,
    learning_rate: float,
    batch_size: int,
    model_parallelism: int,
    save_checkpoints_steps: int,
    n_checkpoints_to_keep: int,
    tpu_name: str,
    tpu_topology: str,
) -> None:
    """Fine-tune the model on MIXTURE, writing results to RESULTS_DIR."""
    utils.configure_logging(clear=True)

    # Validate arguments.

    if not results_dir.startswith("gs://"):
        raise ValueError(f"RESULTS_DIR ({results_dir}) must be a GCS path.")

    if pretrained_model.startswith("gs://"):
        if not tf.io.gfile.exists(pretrained_model):
            raise IOError(
                f"--pretrained-model ({pretrained_model}) does not exist."
            )
    else:
        if pretrained_model not in PRETRAINED_MODELS:
            raise ValueError(
                f"--pretrained-model ({pretrained_model}) not recognized. It"
                f" must either be a GCS path or one of"
                f' {", ".join(PRETRAINED_MODELS.keys())}.'
            )

    # Process arguments.

    if pretrained_model in PRETRAINED_MODELS:
        pretrained_model = PRETRAINED_MODELS[pretrained_model]

    # Run fine-tuning.

    model = t5.models.MtfModel(
        model_dir=results_dir,
        tpu=tpu_name,
        tpu_topology=tpu_topology,
        model_parallelism=model_parallelism,
        batch_size=batch_size,
        sequence_length={"inputs": 512, "targets": 512},
        learning_rate_schedule=learning_rate,
        save_checkpoints_steps=save_checkpoints_steps,
        keep_checkpoint_max=n_checkpoints_to_keep,
        iterations_per_loop=100,
    )

    model.finetune(
        mixture_or_task_name=mixture,
        pretrained_model_dir=pretrained_model,
        finetune_steps=n_steps,
        split=split,
    )


if __name__ == "__main__":
    fine_tune()  # pylint: disable=no-value-for-parameter
