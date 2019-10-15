"""Pre-train RoBERTa on a dataset."""

import math
import json
import logging
import os
import shutil

from apex import amp
import click
import tensorboardX
import torch
from torch.utils.data import DataLoader
import tqdm
import transformers

from rainbow import (
    datasets,
    models,
    settings,
    transforms,
    utils)


logger = logging.getLogger(__name__)


@click.command()
@click.argument(
    'dataset', type=click.Choice(datasets.DATASETS.keys()))
@click.argument(
    'data_dir',
    type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.argument(
    'results_dir',
    type=click.Path(exists=False, file_okay=False, dir_okay=True))
@click.option(
    '--lr', type=float, default=1e-5,
    help='The learning rate for SGD. Defaults to 1e-5.')
@click.option(
    '--weight-decay', type=float, default=1e-2,
    help='The weight decay (l2 regularization penalty). Defaults to 1e-2.')
@click.option(
    '--warmup-proportion', type=float, default=5e-2,
    help='The proportion of iterations to use as linear warmup. Defaults to'
         ' 5e-2.')
@click.option(
    '--n-epochs', type=int, default=1,
    help='The number of epochs for which to run pre-training. Defaults to 1.')
@click.option(
    '--train-batch-size', type=int, default=64,
    help='The batch size to use for training. Defaults to 64.')
@click.option(
    '--compute-train-batch-size', type=int, default=4,
    help='The largest batch size that can fit on the hardware during'
    ' training. Gradient accumulation will be used to make sure the'
    ' actual size of the batch on the hardware respects this'
    ' limit. Defaults to 4.')
@click.option(
    '--predict-batch-size', type=int, default=8,
    help='The batch size for prediction. Defaults to 8.')
@click.option(
    '--opt-level', type=str, default='O0',
    help='The mixed precision optimization level. Defaults to O0. See'
         ' https://nvidia.github.io/apex/amp.html#opt-levels-and-properties'
         ' for more information on the optimization levels.')
@click.option(
    '--augmentation-type',
    type=click.Choice(datasets.AUGMENTATION_TYPES),
    default='original',
    help='The type of augmentation to use on the dataset. Defaults to'
         ' "original".')
@click.option(
    '--pretrained-weights', type=str, default='roberta-large',
    help='The pretrained weights for initializing the model. The pretrained'
         ' weights must be compatible with the roberta-large'
         ' tokenizer. Defaults to "roberta-large".')
@click.option(
    '--gpu-id', type=int, required=True,
    help='The GPU ID to use for training.')
def pre_train(
        dataset: str,
        data_dir: str,
        results_dir: str,
        lr: float,
        weight_decay: float,
        warmup_proportion: float,
        n_epochs: int,
        train_batch_size: int,
        compute_train_batch_size: int,
        predict_batch_size: int,
        opt_level: str,
        augmentation_type: str,
        pretrained_weights: str,
        gpu_id: int
) -> None:
    """Pre-train on DATASET, writing results to RESULTS_DIR."""
    utils.configure_logging()

    # Step 1: Manage and construct paths.
    logger.info('Creating the model directory.')

    checkpoints_dir = os.path.join(results_dir, 'checkpoints')
    tensorboard_dir = os.path.join(results_dir, 'tensorboard')
    os.makedirs(results_dir)
    os.makedirs(checkpoints_dir)
    os.makedirs(tensorboard_dir)

    config_file_path = os.path.join(results_dir, 'config.json')
    log_file_path = os.path.join(results_dir, 'log.txt')
    best_checkpoint_path = os.path.join(checkpoints_dir, 'best.checkpoint.th')
    last_checkpoint_path = os.path.join(checkpoints_dir, 'last.checkpoint.th')

    # Step 2: Setup the log file.
    logger.info('Configuring log files.')

    log_file_handler = logging.FileHandler(log_file_path)
    log_file_handler.setLevel(logging.DEBUG)
    log_file_handler.setFormatter(logging.Formatter(settings.LOG_FORMAT))
    logging.root.addHandler(log_file_handler)

    # Step 3: Record the script's arguments.
    logger.info(f'Writing arguments to {config_file_path}.')

    with open(config_file_path, 'w') as config_file:
        json.dump(
            {
                'dataset': dataset,
                'data_dir': data_dir,
                'results_dir': results_dir,
                'lr': lr,
                'weight_decay': weight_decay,
                'warmup_proporition': warmup_proportion,
                'n_epochs': n_epochs,
                'train_batch_size': train_batch_size,
                'compute_train_batch_size': compute_train_batch_size,
                'predict_batch_size': predict_batch_size,
                'opt_level': opt_level,
                'augmentation_type': augmentation_type,
                'pretrained_weights': pretrained_weights,
                'gpu_id': gpu_id
            },
            config_file)

    # Step 4: Configure GPUs.
    logger.info(f'Configuring environment to use GPU: {gpu_id}.')

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    if not torch.cuda.is_available():
        raise EnvironmentError("CUDA must be available to use GPUs.")
    device = torch.device("cuda")  # pylint: disable=no-member

    # Step 5: Load the dataset.
    logger.info(f'Loading the dataset.')

    # create the train and dev datasets
    tokenizer = transformers.RobertaTokenizer.from_pretrained('roberta-large')
    dataset_class = datasets.DATASETS[dataset]
    transform = transforms.Compose([
        transforms.DistributeContextTransform(),
        transforms.Map(
            transform=transforms.LinearizeTransform(
                tokenizer=tokenizer,
                max_sequence_length=512,
                truncation_strategy='beginning')),
        lambda ds, y: (
            {
                'input_ids': torch.stack([
                    torch.tensor(d['input_ids'])
                    for d in ds
                ], dim=0),
                'attention_mask': torch.stack([
                    torch.tensor(d['input_mask'])
                    for d in ds
                ], dim=0),
            },
            y
        )
    ])
    transform_embedding = lambda e: torch.tensor(e)

    train = dataset_class(
        data_dir=data_dir,
        split='train',
        transform=transform,
        transform_embedding=transform_embedding,
        augmentation_type=augmentation_type)
    dev = dataset_class(
        data_dir=data_dir,
        split='dev',
        transform=transform,
        transform_embedding=transform_embedding,
        augmentation_type=augmentation_type)
    # fit the transform
    transform.fit(train.features, train.labels)

    train_loader = DataLoader(
        dataset=train,
        batch_size=compute_train_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True)
    dev_loader = DataLoader(
        dataset=dev,
        batch_size=predict_batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True)

    # Step 6: Create the model, optimizer, and loss.
    logger.info('Initializing the model.')

    model = models.modeling_roberta.RobertaForMaskedLM.from_pretrained(
        pretrained_weights)
    model.to(device)

    n_gradient_accumulation = math.ceil(
        train_batch_size / compute_train_batch_size)
    n_optimization_steps = n_epochs * math.ceil(len(train) / train_batch_size)

    parameter_groups = [
        {
            'params': [
                param
                for name, param in model.named_parameters()
                if 'bias' in name
                or 'LayerNorm.bias' in name
                or 'LayerNorm.weight' in name
            ],
            'weight_decay': 0
        },
        {
            'params': [
                param
                for name, param in model.named_parameters()
                if 'bias' not in name
                and 'LayerNorm.bias' not in name
                and 'LayerNorm.weight' not in name
            ],
            'weight_decay': weight_decay
        }
    ]
    optimizer = transformers.AdamW(parameter_groups, lr=lr)

    # add fp16 support
    model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)

    scheduler = transformers.WarmupLinearSchedule(
        optimizer=optimizer,
        warmup_steps=int(warmup_proportion * n_optimization_steps),
        t_total=n_optimization_steps)

    loss = torch.nn.CrossEntropyLoss(ignore_index=-1)

    # Step 7: Run training.
    logger.info(f'Running training for {n_epochs} epochs.')

    n_train_batches_per_epoch = math.ceil(len(train) / train_batch_size)
    n_dev_batches_per_epoch = math.ceil(len(dev) / predict_batch_size)

    writer = tensorboardX.SummaryWriter(log_dir=tensorboard_dir)

    best_dev_loss = math.inf
    for epoch in range(n_epochs):
        # set the model to training mode
        model.train()

        # run training for the epoch
        epoch_train_loss = 0
        epoch_train_labels = []
        epoch_train_predictions = []
        for i, (_, features, embeddings, _) in tqdm.tqdm(
            enumerate(train_loader),
            total=n_gradient_accumulation * n_train_batches_per_epoch,
            **settings.TQDM_KWARGS,
        ):
            if augmentation_type not in ['atomic_vector']:
                embeddings = None

            embeddings = (
                embeddings.to(device)
                if embeddings is not None
                else None
            )

            # move the data onto the device
            input_ids = features['input_ids']
            input_ids = input_ids.view(-1, input_ids.shape[-1]).to(device)
            attention_mask = features['attention_mask']
            attention_mask = attention_mask.view(
                -1, attention_mask.shape[-1]).to(device)

            # convert the input_ids into the masked LM features and labels
            #   clone the input_ids into labels before modifying them
            labels = input_ids.clone()
            #   create the mask
            sample_probabilities = torch.full(input_ids.shape, 0.15, device=device)
            sample_probabilities[input_ids == tokenizer.cls_token_id] = 0
            sample_probabilities[input_ids == tokenizer.sep_token_id] = 0
            mask = torch.bernoulli(sample_probabilities).to(torch.uint8)
            #   modify the input_ids
            mask_token_mask = (
                mask
                * torch.bernoulli(
                    torch.full(mask.shape, 0.8, device=device)).to(torch.uint8)
            )
            input_ids[mask_token_mask] = tokenizer.mask_token_id
            random_token_mask = (
                mask
                * (1 - mask_token_mask)
                * torch.bernoulli(
                    torch.full(mask.shape, 0.5, device=device)).to(torch.uint8)
            )
            input_ids[random_token_mask] = torch.randint(
                len(tokenizer),
                input_ids.shape,
                dtype=torch.long,
                device=device)[random_token_mask]
            #   omit the unmasked tokens from being included as labels
            labels[1 - mask] = -1

            # make predictions
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                embeddings=embeddings)[0]
            _, predictions = torch.max(logits, 1)  # pylint: disable=no-member

            batch_loss = loss(
                logits.view(-1, logits.shape[-1]),
                labels.view(-1))

            # update training statistics
            epoch_train_loss = (
                (batch_loss.item() + i * epoch_train_loss)
                / (i + 1)
            )
            epoch_train_labels.extend([
                label
                for label in labels.view(-1).cpu().numpy().tolist()
                if label != -1
            ])
            epoch_train_predictions.extend([
                prediction
                for prediction, label in zip(
                        predictions.view(-1).cpu().numpy().tolist(),
                        labels.view(-1).cpu().numpy().tolist())
                if label != -1
            ])

            # update the network
            with amp.scale_loss(batch_loss, optimizer) as scaled_loss:
                scaled_loss.backward()

            if (i + 1) % n_gradient_accumulation == 0:
                optimizer.step()
                optimizer.zero_grad()

                scheduler.step()

            # write training statistics to tensorboard

            step = n_train_batches_per_epoch * epoch + (
                (i + 1) // n_gradient_accumulation
            )
            if step % 50 == 0 and (i + 1) % n_gradient_accumulation == 0:
                epoch_train_score = dataset_class.metric(
                    y_true=epoch_train_labels,
                    y_pred=epoch_train_predictions)
                writer.add_scalar('train/loss', epoch_train_loss, step)
                writer.add_scalar('train/score', epoch_train_score, step)

        # compute the final train score for the epoch
        epoch_train_score = dataset_class.metric(
            y_true=epoch_train_labels,
            y_pred=epoch_train_predictions)

        # run evaluation
        with torch.no_grad():
            # set the model to evaluation mode
            model.eval()

            # run validation for the epoch
            epoch_dev_loss = 0
            epoch_dev_labels = []
            epoch_dev_predictions = []
            for i, (_, features, embeddings, _) in tqdm.tqdm(
                enumerate(dev_loader),
                total=n_dev_batches_per_epoch,
                **settings.TQDM_KWARGS,
            ):
                if augmentation_type not in ['atomic_vector']:
                    embeddings = None

                embeddings = (
                    embeddings.to(device)
                    if embeddings is not None
                    else None
                )

                # move the data onto the device
                input_ids = features['input_ids']
                input_ids = input_ids.view(-1, input_ids.shape[-1]).to(device)
                attention_mask = features['attention_mask']
                attention_mask = attention_mask.view(-1, attention_mask.shape[-1]).to(device)

                # convert the input_ids into the masked LM features
                #   clone the input_ids into labels before modifying them
                labels = input_ids.clone()
                #   create the mask
                sample_probabilities = torch.full(input_ids.shape, 0.15, device=device)
                sample_probabilities[input_ids == tokenizer.cls_token_id] = 0
                sample_probabilities[input_ids == tokenizer.sep_token_id] = 0
                mask = torch.bernoulli(sample_probabilities).to(torch.uint8)
                #   modify the input_ids
                mask_token_mask = (
                    mask
                    * torch.bernoulli(
                        torch.full(mask.shape, 0.8, device=device)).to(torch.uint8)
                )
                input_ids[mask_token_mask] = tokenizer.mask_token_id
                random_token_mask = (
                    mask
                    * (1 - mask_token_mask)
                    * torch.bernoulli(
                        torch.full(mask.shape, 0.5, device=device)).to(torch.uint8)
                )
                input_ids[random_token_mask] = torch.randint(
                    len(tokenizer),
                    input_ids.shape,
                    dtype=torch.long,
                    device=device)[random_token_mask]
                #   omit the unmasked tokens from being included as labels
                labels = input_ids.clone()

                # make predictions
                logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    embeddings=embeddings)[0]
                _, predictions = torch.max(logits, 1)  # pylint: disable=no-member

                batch_loss = loss(
                    logits.view(-1, logits.shape[-1]),
                    labels.view(-1))

                # update validation statistics
                epoch_dev_loss = (
                    (batch_loss.item() + i * epoch_dev_loss)
                    / (i + 1)
                )
                epoch_dev_labels.extend([
                    label
                    for label in labels.view(-1).cpu().numpy().tolist()
                    if label != -1
                ])
                epoch_dev_predictions.extend([
                    prediction
                    for prediction, label in zip(
                            predictions.view(-1).cpu().numpy().tolist(),
                            labels.view(-1).cpu().numpy().tolist())
                    if label != -1
                ])

            # write validation statistics to tensorboard
            epoch_dev_score = dataset_class.metric(
                y_true=epoch_dev_labels,
                y_pred=epoch_dev_predictions)
            writer.add_scalar('dev/loss', epoch_dev_loss, step)
            writer.add_scalar('dev/score', epoch_dev_score, step)

            logger.info(
                f'\n\n'
                f'  epoch {epoch}:\n'
                f'    train loss  : {epoch_train_loss:.4f}\n'
                f'    train score : {epoch_train_score:.4f}\n'
                f'    dev loss    : {epoch_dev_loss:.4f}\n'
                f'    dev score   : {epoch_dev_score:.4f}\n'
            )

        # update checkpoints

        torch.save(
            {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            },
            last_checkpoint_path)

        # update the current best model
        if epoch_dev_loss < best_dev_loss:
            shutil.copyfile(last_checkpoint_path, best_checkpoint_path)
            best_dev_loss = epoch_dev_loss

        # exit early if the training loss has diverged
        if math.isnan(epoch_train_loss):
            logger.info('Training loss has diverged. Exiting early.')
            return

    logger.info('Saving the best checkpoint as a pre-trained model.')

    model.load_state_dict(torch.load(best_checkpoint_path)['model'])
    model.save_pretrained(results_dir)

    logger.info(f'Training complete. Best dev loss was {best_dev_loss:.4f}')


if __name__ == '__main__':
    pre_train()
