"""Mixture definitions for rainbow."""

import t5

from . import core, datasets, settings, tasks

# N.B. tasks must be imported before mixtures, so that the mixtures can use
# the tasks in their definitions.


# Create the individual rainbow task mixtures.
for dataset in datasets.RAINBOW_DATASETS.values():
    for size in settings.LEARNING_CURVE_SIZES:
        base_name = (
            f"{dataset.name}" if size is None else f"{dataset.name}_{size:05}"
        )
        t5.data.MixtureRegistry.add(
            f"{base_name}_mixture", [f"{base_name}_task"], default_rate=1.0
        )

# Create the individual knowledge graph task mixtures
for dataset in datasets.KNOWLEDGE_GRAPH_DATASETS.values():
    for size in settings.LEARNING_CURVE_SIZES:
        for direction in settings.KNOWLEDGE_GRAPH_DIRECTIONS:
            base_name = (
                f"{dataset.name}_{direction}"
                if size is None
                else f"{dataset.name}_{direction}_{size:05}"
            )
            t5.data.MixtureRegistry.add(
                f"{base_name}_mixture", [f"{base_name}_task"], default_rate=1.0
            )

# Create the Rainbow mixtures.
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

# Create mixtures with rainbow and a single knowledge graph.
for dataset in datasets.KNOWLEDGE_GRAPH_DATASETS.values():
    for direction in settings.KNOWLEDGE_GRAPH_DIRECTIONS:
        t5.data.MixtureRegistry.add(
            f"rainbow_{dataset.name}_{direction}_equal_mixture",
            [f"{dataset.name}_{direction}_task"]
            + [
                f"{rainbow_dataset.name}_task"
                for rainbow_dataset in datasets.RAINBOW_DATASETS.values()
            ],
            default_rate=core.equal_rate,
        )

        t5.data.MixtureRegistry.add(
            f"rainbow_{dataset.name}_{direction}_proportional_mixture",
            [f"{dataset.name}_{direction}_task"]
            + [
                f"{rainbow_dataset.name}_task"
                for rainbow_dataset in datasets.RAINBOW_DATASETS.values()
            ],
            default_rate=core.proportional_rate,
        )

# Create mixtures with rainbow and all knowledge graphs.
for direction in settings.KNOWLEDGE_GRAPH_DIRECTIONS:
    t5.data.MixtureRegistry.add(
        f"rainbow_comet_{direction}_equal_mixture",
        [
            # include the knowledge graphs
            f"{dataset.name}_{direction}_task"
            for dataset in datasets.KNOWLEDGE_GRAPH_DATASETS.values()
        ]
        + [
            # include the rainbow datasets
            f"{dataset.name}_task"
            for dataset in datasets.RAINBOW_DATASETS.values()
        ],
        default_rate=core.equal_rate,
    )

    t5.data.MixtureRegistry.add(
        f"rainbow_comet_{direction}_proportional_mixture",
        [
            # include the knowledge graphs
            f"{dataset.name}_{direction}_task"
            for dataset in datasets.KNOWLEDGE_GRAPH_DATASETS.values()
        ]
        + [
            # include the rainbow datasets
            f"{dataset.name}_task"
            for dataset in datasets.RAINBOW_DATASETS.values()
        ],
        default_rate=core.proportional_rate,
    )

# Create the Rainbow multi-tasking learning curve mixtures.
for dataset in datasets.RAINBOW_DATASETS.values():
    for size in settings.LEARNING_CURVE_SIZES:
        if size is None:
            continue

        t5.data.MixtureRegistry.add(
            f"{dataset.name}_{size:05}_rainbow_equal_mixture",
            [f"{dataset.name}_{size:05}_task"]
            + [
                f"{other_dataset.name}_task"
                for other_dataset in datasets.RAINBOW_DATASETS.values()
                if other_dataset != dataset
            ],
            default_rate=core.equal_rate,
        )

        t5.data.MixtureRegistry.add(
            f"{dataset.name}_{size:05}_rainbow_proportional_mixture",
            [f"{dataset.name}_{size:05}_task"]
            + [
                f"{other_dataset.name}_task"
                for other_dataset in datasets.RAINBOW_DATASETS.values()
                if other_dataset != dataset
            ],
            default_rate=core.proportional_rate,
        )

# Create the GLUE multi-tasking learning curve mixtures.
for dataset in datasets.RAINBOW_DATASETS.values():
    for size in settings.LEARNING_CURVE_SIZES:
        if size is None:
            continue

        t5.data.MixtureRegistry.add(
            f"{dataset.name}_{size:05}_glue_equal_mixture",
            [f"{dataset.name}_{size:05}_task"]
            + [
                f"glue_{glue_dataset.name}_v002"
                for glue_dataset in datasets.GLUE_DATASETS.values()
            ],
            default_rate=core.equal_rate,
        )

        t5.data.MixtureRegistry.add(
            f"{dataset.name}_{size:05}_glue_proportional_mixture",
            [f"{dataset.name}_{size:05}_task"]
            + [
                f"glue_{glue_dataset.name}_v002"
                for glue_dataset in datasets.GLUE_DATASETS.values()
            ],
            default_rate=core.proportional_rate,
        )

# Create the Super GLUE multi-tasking learning curve mixtures.
for dataset in datasets.RAINBOW_DATASETS.values():
    for size in settings.LEARNING_CURVE_SIZES:
        if size is None:
            continue

        t5.data.MixtureRegistry.add(
            f"{dataset.name}_{size:05}_super_glue_equal_mixture",
            [f"{dataset.name}_{size:05}_task"]
            + [
                f"super_glue_{super_glue_dataset.name}_v102"
                for super_glue_dataset in datasets.SUPER_GLUE_DATASETS.values()
            ],
            default_rate=core.equal_rate,
        )

        t5.data.MixtureRegistry.add(
            f"{dataset.name}_{size:05}_super_glue_proportional_mixture",
            [f"{dataset.name}_{size:05}_task"]
            + [
                f"super_glue_{super_glue_dataset.name}_v102"
                for super_glue_dataset in datasets.SUPER_GLUE_DATASETS.values()
            ],
            default_rate=core.proportional_rate,
        )
