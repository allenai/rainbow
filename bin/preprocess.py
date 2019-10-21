"""Preprocess a dataset to augment it with additional features."""

import logging
import os

import attr
import click
import msgpack

from rainbow import (
    comet,
    datasets,
    features)


logger = logging.getLogger(__name__)


@click.command()
@click.argument(
    'dataset', type=click.Choice(datasets.DATASETS.keys()))
@click.argument(
    'dataset_path',
    type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.argument(
    'atomic_comet_model_path',
    type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.argument(
    'atomic_comet_vocab_path',
    type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.argument(
    'conceptnet_comet_model_path',
    type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.argument(
    'conceptnet_comet_vocab_path',
    type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.argument(
    'output_dir',
    type=click.Path(exists=False, file_okay=False, dir_okay=True))
@click.option(
    '--batch-size', type=int, default=128,
    help='The batch size to use for generating COMeT predictions.')
def preprocess(
        dataset: str,
        dataset_path: str,
        atomic_comet_model_path: str,
        atomic_comet_vocab_path: str,
        conceptnet_comet_model_path: str,
        conceptnet_comet_vocab_path: str,
        output_dir: str,
        batch_size: int
) -> None:
    """Preprocess DATASET writing results to OUTPUT_DIR.

    Read DATASET from DATASET_PATH, preprocess it by running all possible
    feature augmentations on it, and write the results to OUTPUT_DIR.
    """
    os.makedirs(output_dir)

    dataset_class = datasets.DATASETS[dataset]
    for split in ['train', 'dev']:
        # process the split

        logger.info(f'Augmenting {split}.')

        # load the data
        instances = dataset_class.read_raw_instances(
            dataset_path=dataset_path,
            split=split)

        # check that all the instances have the same feature names
        feature_names = set(instances[0].features.keys())
        for i, instance in enumerate(instances):
            if set(instance.features.keys()) != feature_names:
                raise ValueError(
                    f'instance {i} in split {split} has inconsistent feature'
                    f' names.')

        # write out the original data in the multiple choice format

        logger.info(f'Writing {split} without any augmentation.')

        unaugmented_path = os.path.join(
            output_dir,
            dataset_class.preprocessed_path_templates['original'].format(
                split=split,
                name=dataset_class.name))
        with open(unaugmented_path, 'wb') as unaugmented_file:
            for i, instance in enumerate(instances):
                unaugmented_file.write(msgpack.packb(
                    attr.asdict(instance),
                    use_bin_type=True))

        # augment the batches with ATOMIC

        logger.info(f'Augmenting {split} with COMeT ATOMIC features.')

        # compute the features
        atomic_features = {
            feature_name: comet.augment_with_atomic_comet(
                features=[
                    instance.features[feature_name]
                    for instance in instances
                ],
                model_path=atomic_comet_model_path,
                vocab_path=atomic_comet_vocab_path,
                batch_size=batch_size)
            for feature_name in feature_names
        }
        atomic_answer_features = [
            comet.augment_with_atomic_comet(
                features=[
                    instance.answers[i]
                    for instance in instances
                ],
                model_path=atomic_comet_model_path,
                vocab_path=atomic_comet_vocab_path,
                batch_size=batch_size)
            for i in range(dataset_class.num_choices)
        ]

        # write the features to disk
        atomic_augmented_path = os.path.join(
            output_dir,
            dataset_class.preprocessed_path_templates['atomic'].format(
                split=split,
                name=dataset_class.name))
        with open(atomic_augmented_path, 'wb') as atomic_augmented_file:
            for i, instance in enumerate(instances):
                instance = attr.evolve(
                    instance,
                    features={
                        feature_name: atomic_features[feature_name][i]
                        for feature_name in feature_names
                    },
                    answers=[
                        atomic_answer_features[j][i]
                        for j in range(dataset_class.num_choices)
                    ])
                atomic_augmented_file.write(msgpack.packb(
                    attr.asdict(instance),
                    use_bin_type=True))

        # augment the batches with ConceptNet

        logger.info(f'Augmenting {split} with COMeT ConceptNet features.')

        # compute the features
        conceptnet_features = {
            feature_name: comet.augment_with_conceptnet_comet(
                features=[
                    instance.features[feature_name]
                    for instance in instances
                ],
                model_path=conceptnet_comet_model_path,
                vocab_path=conceptnet_comet_vocab_path,
                batch_size=batch_size)
            for feature_name in feature_names
        }
        conceptnet_answer_features = [
            comet.augment_with_conceptnet_comet(
                features=[
                    instance.answers[i]
                    for instance in instances
                ],
                model_path=conceptnet_comet_model_path,
                vocab_path=conceptnet_comet_vocab_path,
                batch_size=batch_size)
            for i in range(dataset_class.num_choices)
        ]

        # write the features to disk
        conceptnet_augmented_path = os.path.join(
            output_dir,
            dataset_class.preprocessed_path_templates['conceptnet'].format(
                split=split,
                name=dataset_class.name))
        with open(conceptnet_augmented_path, 'wb') as conceptnet_augmented_file:
            for i, instance in enumerate(instances):
                instance = attr.evolve(
                    instance,
                    features={
                        feature_name: conceptnet_features[feature_name][i]
                        for feature_name in feature_names
                    },
                    answers=[
                        conceptnet_answer_features[j][i]
                        for j in range(dataset_class.num_choices)
                    ])
                conceptnet_augmented_file.write(msgpack.packb(
                    attr.asdict(instance),
                    use_bin_type=True))

    logger.info('Preprocessing is finished.')


if __name__ == '__main__':
    preprocess()
