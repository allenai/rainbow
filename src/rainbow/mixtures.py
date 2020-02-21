"""Mixture definitions for rainbow."""

import t5

from . import core, datasets, settings, tasks

# N.B. tasks must be imported before mixtures, so that the mixtures can use
# the tasks in their definitions.


# Create the individual task mixtures

for dataset in datasets.RAINBOW_DATASETS.values():
    t5.data.MixtureRegistry.add(
        f"{dataset.name}_mixture", [f"{dataset.name}_task"], default_rate=1.0
    )


# Create the leave-one-task-out mixtures

for dataset in datasets.RAINBOW_DATASETS.values():
    for size in settings.LEARNING_CURVE_SIZES:
        if size is None:
            continue

        t5.data.MixtureRegistry.add(
            f"{dataset.name}_{size}_rainbow_equal_mixture",
            [f"{dataset.name}_{size}_task"]
            + [
                f"{other_dataset.name}_task"
                for other_dataset in datasets.RAINBOW_DATASETS.values()
                if other_dataset != dataset
            ],
            default_rate=core.equal_rate,
        )

        t5.data.MixtureRegistry.add(
            f"{dataset.name}_{size}_rainbow_proportional_mixture",
            [f"{dataset.name}_{size}_task"]
            + [
                f"{other_dataset.name}_task"
                for other_dataset in datasets.RAINBOW_DATASETS.values()
                if other_dataset != dataset
            ],
            default_rate=core.proportional_rate,
        )

# Create the full Rainbow mixtures

t5.data.MixtureRegistry.add(
    "rainbow_equal_mixture",
    [f"{dataset.name}_task" for dataset in datasets.RAINBOW_DATASETS.values()],
    default_rate=core.equal_rate,
)

t5.data.MixtureRegistry.add(
    "rainbow_proportional_mixture",
    [f"{dataset.name}_task" for dataset in datasets.RAINBOW_DATASETS.values()],
    default_rate=core.proportional_rate,
)
