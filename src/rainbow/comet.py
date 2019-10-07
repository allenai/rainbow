"""COMeT generations and representations."""

import math
from typing import List

from comet.data import config as cfg
from comet.data.data import atomic_data, conceptnet_data, end_token as END_TOKEN
from comet.models.utils import prepare_position_embeddings
from comet.interactive import functions as comet
import torch
import tqdm

from rainbow.features import (
    TextFeature,
    AtomicCometFeature,
    ConceptNetCometFeature,
)
from rainbow.utils import transpose_dictionary


ATOMIC_CATEGORIES = [
    ("o_effect", "oEffect"),
    ("o_react", "oReact"),
    ("o_want", "oWant"),
    ("x_attr", "xAttr"),
    ("x_effect", "xEffect"),
    ("x_intent", "xIntent"),
    ("x_need", "xNeed"),
    ("x_react", "xReact"),
    ("x_want", "xWant"),
]

CONCEPTNET_CATEGORIES = [
    ("at_location", "AtLocation"),
    ("capable_of", "CapableOf"),
    ("causes", "Causes"),
    ("causes_desire", "CausesDesire"),
    ("created_by", "CreatedBy"),
    ("defined_as", "DefinedAs"),
    ("desire_of", "DesireOf"),
    ("desires", "Desires"),
    ("has_a", "HasA"),
    ("has_first_subevent", "HasFirstSubevent"),
    ("has_last_subevent", "HasLastSubevent"),
    ("has_pain_character", "HasPainCharacter"),
    ("has_pain_intensity", "HasPainIntensity"),
    ("has_prerequisite", "HasPrerequisite"),
    ("has_property", "HasProperty"),
    ("has_subevent", "HasSubevent"),
    ("inherits_from", "InheritsFrom"),
    ("instance_of", "InstanceOf"),
    ("is_a", "IsA"),
    ("located_near", "LocatedNear"),
    ("location_of_action", "LocationOfAction"),
    ("made_of", "MadeOf"),
    ("motivated_by_goal", "MotivatedByGoal"),
    ("not_capable_of", "NotCapableOf"),
    ("not_desires", "NotDesires"),
    ("not_has_a", "NotHasA"),
    ("not_has_property", "NotHasProperty"),
    ("not_is_a", "NotIsA"),
    ("not_made_of", "NotMadeOf"),
    ("part_of", "PartOf"),
    ("receives_action", "ReceivesAction"),
    ("related_to", "RelatedTo"),
    ("symbol_of", "SymbolOf"),
    ("used_for", "UsedFor"),
]


def decode(sequence, data_loader):
    """Return the sequence decoded into text."""
    end_token_idx = data_loader.vocab_encoder[END_TOKEN]

    word_pieces = []
    for index in sequence:
        if index == end_token_idx:
            break
        word_pieces.append(
            data_loader.vocab_decoder[index.item()].replace("</w>", " ")
        )

    return "".join(word_pieces).strip()


def _generate_samples(
    mb_xs, mb_mask, model, data_loader, end_len
) -> List[List[str]]:
    """Return greedily decoded generations."""
    sequences = (
        torch.zeros([mb_xs.shape[0], 0])  # pylint: disable=no-member
        .long()
        .to(cfg.device)
    )
    for _ in range(end_len):
        logits = model(mb_xs, sequence_mask=mb_mask)
        _, indices = logits[:, -1:, :].max(dim=-1)

        # add the predicted next tokens to the output sequences
        sequences = torch.cat(  # pylint: disable=no-member
            [sequences, indices], dim=-1
        )

        # add the predicted next tokens to the features / masks
        next_pos = mb_xs[:, -1:, 1] + 1
        next_xs = torch.stack(  # pylint: disable=no-member
            [indices, next_pos], dim=2
        )
        next_mask = torch.ones(  # pylint: disable=no-member
            [mb_xs.shape[0], 1]
        ).to(cfg.device)

        mb_xs = torch.cat([mb_xs, next_xs], dim=1)  # pylint: disable=no-member
        mb_mask = torch.cat(  # pylint: disable=no-member
            [mb_mask, next_mask], dim=1
        )

    samples = [decode(sequence, data_loader) for sequence in sequences]

    return samples


def augment_with_atomic_comet(
    features: List[TextFeature],
    model_path: str,
    vocab_path: str,
    batch_size: int,
) -> List[AtomicCometFeature]:
    """Return ``features`` converted into ATOMIC COMeT features.

    Parameters
    ----------
    features : List[TextFeature]
        A list of ``TextFeature`` instances.
    model_path : str
        The path to the ATOMIC COMeT model.
    vocab_path : str
        The path to the vocabulary for the ATOMIC COMeT model.
    batch_size : int
        The batch size to use for prediction when generating from COMeT.

    Returns
    -------
    List[AtomicCometFeature]
        The ``TextFeature`` instances in ``features`` converted into
        ``AtomicCometFeature`` instances.
    """
    # load the COMeT components
    opt, state_dict, vocab = comet.load_model_file(model_path)
    data_loader, text_encoder = comet.load_data(
        "atomic", opt, vocab, vocab_path
    )

    n_ctx = data_loader.max_event + data_loader.max_effect
    n_vocab = len(text_encoder.encoder) + n_ctx

    model = comet.make_model(opt, n_vocab, n_ctx, state_dict)

    # run batched predictions on the data
    atomic_augmented_features = []
    for i in tqdm.tqdm(
        range(0, len(features), batch_size),
        total=math.ceil(len(features) / batch_size),
        ncols=79,
        leave=False,
    ):
        with torch.no_grad():
            mb_texts = [
                feature.text for feature in features[i : i + batch_size]
            ]

            # make a prediction for each category
            attr_name_to_augmented_features = {"text": mb_texts}
            for attr_name, category in ATOMIC_CATEGORIES:
                mb_xs = torch.zeros(  # pylint: disable=no-member
                    batch_size, data_loader.max_event + 1
                ).long()
                for j, text in enumerate(mb_texts):
                    encoded_text = torch.LongTensor(  # pylint: disable=no-member
                        text_encoder.encode([text], verbose=False)[0]
                    )[
                        : data_loader.max_event
                    ]
                    mb_xs[j, : len(encoded_text)] = encoded_text
                mb_xs[:, -1] = torch.LongTensor(  # pylint: disable=no-member
                    [text_encoder.encoder[f"<{category}>"]]
                )
                mb_mask = (mb_xs != 0).float()

                mb_xs = mb_xs.to(cfg.device)
                mb_mask = mb_mask.to(cfg.device)

                mb_xs = prepare_position_embeddings(
                    opt, data_loader.vocab_encoder, mb_xs.unsqueeze(-1)
                )

                representations = model.transformer(mb_xs, mb_mask)[
                    :, data_loader.max_event
                ]
                # N.B. Grab the hidden representation over the relation token
                # as the COMeT representation.
                samples = _generate_samples(
                    mb_xs,
                    mb_mask,
                    model=model,
                    data_loader=data_loader,
                    end_len=data_loader.max_effect - 1,
                )

                attr_name_to_augmented_features[attr_name] = samples
                attr_name_to_augmented_features[f"{attr_name}_embeddings"] = (
                    representations.cpu().numpy().tolist()
                )

        for atomic_augmented_feature_kwargs in transpose_dictionary(
            attr_name_to_augmented_features
        ):
            atomic_augmented_features.append(
                AtomicCometFeature(**atomic_augmented_feature_kwargs)
            )

    return atomic_augmented_features


def augment_with_conceptnet_comet(
    features: List[TextFeature],
    model_path: str,
    vocab_path: str,
    batch_size: int,
) -> List[ConceptNetCometFeature]:
    """Return ``features`` converted into ConceptNet COMeT features.

    Parameters
    ----------
    features : List[TextFeature]
        A list of ``TextFeature`` instances.
    model_path : str
        The path to the ConceptNet COMeT model.
    vocab_path : str
        The path to the vocabulary for the ConceptNet COMeT model.
    batch_size : int
        The batch size to use for prediction when generating from COMeT.

    Returns
    -------
    List[ConceptNetCometFeature]
        The ``TextFeature`` instances in ``features`` converted into
        ``ConceptNetCometFeature`` instances.
    """
    # load the COMeT components
    opt, state_dict, vocab = comet.load_model_file(model_path)
    data_loader, text_encoder = comet.load_data(
        "conceptnet", opt, vocab, vocab_path
    )

    n_ctx = data_loader.max_e1 + data_loader.max_r + data_loader.max_e2
    n_vocab = len(text_encoder.encoder) + n_ctx

    model = comet.make_model(opt, n_vocab, n_ctx, state_dict)

    # run batched predictions on the data
    conceptnet_augmented_features = []
    for i in tqdm.tqdm(
        range(0, len(features), batch_size),
        total=math.ceil(len(features) / batch_size),
        ncols=79,
        leave=False,
    ):
        with torch.no_grad():
            mb_texts = [
                feature.text for feature in features[i : i + batch_size]
            ]

            # make a prediction for each category
            attr_name_to_augmented_features = {"text": mb_texts}
            for attr_name, category in CONCEPTNET_CATEGORIES:
                mb_xs = torch.zeros(  # pylint: disable=no-member
                    batch_size, data_loader.max_e1 + data_loader.max_r
                ).long()
                for j, text in enumerate(mb_texts):
                    encoded_text = torch.LongTensor(  # pylint: disable=no-member
                        text_encoder.encode([text], verbose=False)[0]
                    )[
                        : data_loader.max_e1
                    ]
                    mb_xs[j, : len(encoded_text)] = encoded_text
                relation_tokens = text_encoder.encode(
                    [conceptnet_data.split_into_words[category]], verbose=False
                )[0]
                mb_xs[
                    :,
                    data_loader.max_e1 : data_loader.max_e1
                    + len(relation_tokens),
                ] = torch.LongTensor(  # pylint: disable=no-member
                    [relation_tokens]
                )
                mb_mask = (mb_xs != 0).float()

                mb_xs = mb_xs.to(cfg.device)
                mb_mask = mb_mask.to(cfg.device)

                mb_xs = prepare_position_embeddings(
                    opt, data_loader.vocab_encoder, mb_xs.unsqueeze(-1)
                )

                representations = model.transformer(mb_xs, mb_mask)[
                    :, data_loader.max_e1
                ]
                # N.B. Grab the hidden representation over the relation token
                # as the COMeT representation.
                samples = _generate_samples(
                    mb_xs,
                    mb_mask,
                    model=model,
                    data_loader=data_loader,
                    end_len=data_loader.max_e2 - 1,
                )

                attr_name_to_augmented_features[attr_name] = samples
                attr_name_to_augmented_features[f"{attr_name}_embeddings"] = (
                    representations.cpu().numpy().tolist()
                )

        for conceptnet_augmented_feature_kwargs in transpose_dictionary(
            attr_name_to_augmented_features
        ):
            conceptnet_augmented_features.append(
                ConceptNetCometFeature(**conceptnet_augmented_feature_kwargs)
            )

    return conceptnet_augmented_features
