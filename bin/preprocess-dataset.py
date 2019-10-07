"""Preprocess a dataset to augment it with additional features."""

import json
import logging
import os

import attr
import click

from rainbow import (
    comet,
    datasets,
    features)


logger = logging.getLogger(__name__)


DATASETS = {
    'socialiqa': datasets.SocialIQADataset
}


@click.command()
@click.argument(
    'dataset', type=click.Choice(DATASETS.keys()))
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
def preprocess_dataset(
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

    dataset_class = DATASETS[dataset]
    for split in ['train', 'dev']:
        # process the split

        logger.info(f'Augmenting {split}.')

        # load the data
        data = dataset_class(dataset_path=dataset_path, split=split)

        # check that all the instances have the same feature names
        feature_names = set(data.instances[0].features.keys())
        for i, instance in enumerate(data.instances):
            if set(instance.features.keys()) != feature_names:
                raise ValueError(
                    f'instance {i} in split {split} has inconsistent feature'
                    f' names.')

        # augment the batches with ATOMIC

        logger.info(f'Augmenting {split} with COMeT ATOMIC features.')

        # compute the features
        atomic_features = {
            feature_name: comet.augment_with_atomic_comet(
                features=[
                    instance.features[feature_name]
                    for instance in data.instances
                ],
                model_path=atomic_comet_model_path,
                vocab_path=atomic_comet_vocab_path,
                batch_size=batch_size)
            for feature_name in feature_names
        }

        # write the features to disk
        atomic_augmented_path = os.path.join(
            output_dir, f'{split}.atomic-{dataset}.jsonl')
        with open(atomic_augmented_path, 'w') as atomic_augmented_file:
            for i, instance in enumerate(data.instances):
                instance = attr.evolve(
                    instance,
                    features={
                        feature_name: atomic_features[feature_name][i]
                        for feature_name in feature_names
                    })
                atomic_augmented_file.write(
                    json.dumps(attr.asdict(instance)) + '\n')

        # augment the batches with ConceptNet

        logger.info(f'Augmenting {split} with COMeT ConceptNet features.')

        # compute the features
        conceptnet_features = {
            feature_name: comet.augment_with_conceptnet_comet(
                features=[
                    instance.features[feature_name]
                    for instance in data.instances
                ],
                model_path=conceptnet_comet_model_path,
                vocab_path=conceptnet_comet_vocab_path,
                batch_size=batch_size)
            for feature_name in feature_names
        }

        # write the features to disk
        conceptnet_augmented_path = os.path.join(
            output_dir, f'{split}.conceptnet-{dataset}.jsonl')
        with open(conceptnet_augmented_path, 'w') as conceptnet_augmented_file:
            for i, instance in enumerate(data.instances):
                instance = attr.evolve(
                    instance,
                    features={
                        feature_name: conceptnet_features[feature_name][i]
                        for feature_name in feature_names
                    })
                conceptnet_augmented_file.write(
                    json.dumps(attr.asdict(instance)) + '\n')

    logger.info('Preprocessing is finished.')


if __name__ == '__main__':
    preprocess_dataset()
