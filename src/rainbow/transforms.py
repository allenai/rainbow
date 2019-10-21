"""Transformations for reading the data."""

from typing import Any, Callable, Dict, List, Sequence, Tuple, Union

import numpy as np
from transformers import RobertaTokenizer


class Map(object):
    """Map a transform across a sequence.

    Parameters
    ----------
    transform : Callable
        The transform to map across the sequence.
    """

    def __init__(self, transform: Callable) -> None:
        self.transform = transform

        self._fitted = False

    def fit(self, xs: Sequence[Sequence[Any]], ys: Sequence[Any]) -> None:
        if hasattr(self.transform, "fit"):
            self.transform.fit([x_part for x in xs for x_part in x], ys)

        self._fitted = True

    def __call__(self, x: Sequence[Any], y: Any) -> Tuple[List[Any], Any]:
        """Return ``(x, y)`` with ``x`` mapped by ``self.transform``.

        Returns
        -------
        List[Any]
            ``x`` mapped by the transform with which this object was
            initialized, and then cast to a list.
        Any
            ``y``.
        """
        if not self._fitted:
            raise ValueError(
                "The transform must be fitted before it can be called."
            )

        return [self.transform(x_part, y)[0] for x_part in x], y


class Compose(object):
    """Compose a sequence of transforms.

    Parameters
    ----------
    transforms : Sequence[Callable]
        The sequence of transforms to compose. The transforms should be
        in pipeline order, i.e. the first transform to apply goes first
        in the list.
    """

    def __init__(self, transforms: Sequence[Callable]) -> None:
        self.transforms = transforms

        self._fitted = False

    def fit(self, xs: Sequence[Any], ys: Sequence[Any]) -> None:
        self._fitted = True

        if len(xs) == 0 and len(ys) == 0:
            return

        for transform in self.transforms:
            if hasattr(transform, "fit"):
                transform.fit(xs, ys)
            xs, ys = zip(*[transform(x, y) for x, y in zip(xs, ys)])

    def __call__(self, x: Any, y: Any) -> Tuple[Any, Any]:
        """Return ``(x, y)`` with all the transforms applied.

        Returns
        -------
        Any
            The result of applying all the transforms to ``x``.
        Any
            The result of applying all the transforms to ``y``.
        """
        if not self._fitted:
            raise ValueError(
                "The transform must be fitted before it can be called."
            )

        for transform in self.transforms:
            x, y = transform(x, y)

        return x, y


class LinearizeTransform(object):
    """Transform a tuple of text into input for RoBERTa.

    Parameters
    ----------
    tokenizer : RobertaTokenizer, required
        The ``RobertaTokenizer`` instance to use for tokenization.
    max_sequence_length : int, optional (default=512)
        The maximum length the output sequence should be. If the input
        converts to something longer, it will be truncated. The max
        sequence length must be shorter than the maximum sequence length
        the model accepts (512 for the standard pretrained RoBERTa).
    truncation_strategy : Union[str, Sequence[str]],
                          optional (default="beginning")
        A string or sequence of strings providing the truncation
        strategies to use on each piece of text. Each string must be one
        of ``"beginning"``, or ``"ending"``. These strings correspond to
        taking text from the beginning of the input, or the ending of
        the input, respectively.
    """

    TRUNCATION_STRATEGIES = ["beginning", "ending"]

    def __init__(
        self,
        tokenizer: RobertaTokenizer,
        max_sequence_length: int = 512,
        truncation_strategy: Union[str, Sequence[str]] = "beginning",
    ) -> None:
        if not isinstance(tokenizer, RobertaTokenizer):
            raise ValueError("tokenizer must be a type of RobertaTokenizer")

        max_sequence_length = int(max_sequence_length)
        if max_sequence_length < 0:
            raise ValueError(
                "max_sequence_length must be greater than or equal to 0"
            )

        if isinstance(truncation_strategy, str):
            truncation_strategies = [truncation_strategy]
        else:
            truncation_strategies = truncation_strategy
        for strategy in truncation_strategies:
            if strategy not in self.TRUNCATION_STRATEGIES:
                raise ValueError(
                    f"truncation strategy {strategy} does not"
                    f" exist. Please use one of"
                    f" {', '.join(self.TRUNCATION_STRATEGIES)}."
                )

        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length
        self.truncation_strategy = truncation_strategy

        self._fitted = False

    def fit(self, xs: Sequence[Sequence[str]], ys: Sequence[Any]) -> None:
        """Fit this transformation to ``xs`` and ``ys``.

        Fit ``max_sequence_length`` to be as small as possible without causing
        additional truncation of any instances in ``xs`` as tokenized by
        ``tokenizer``.

        Parameters
        ----------
        xs : Sequence[Sequence[str]]
            A sequence of sequences giving all the instances over which
            you wish to compute the max sequence length.
        ys : Sequence[Any]
            The labels.
        """
        self._fitted = True

        min_max_sequence_length = 0
        for x, y in zip(xs, ys):
            min_max_sequence_length = max(
                min_max_sequence_length, np.sum(self(x, y)[0]["input_mask"])
            )
        self.max_sequence_length = min_max_sequence_length

    def __call__(
        self, x: Sequence[str], y: Any
    ) -> Tuple[Dict[str, List[int]], Any]:
        """Return ``(x, y)`` with ``x`` linearized into token ids.

        Parameters
        ----------
        x : Sequence[str]
            The sequence of text to transform into linearized inputs.
        y : Any
            The label.

        Returns
        -------
        Dict[str, List[int]]
            input_ids : List[int]
                The list of IDs for the tokenized word pieces.
            input_mask : List[int]
                A 0-1 mask for ``input_ids``.
        Any
           The label.
        """
        if not self._fitted:
            raise ValueError(
                "The transform must be fitted before it can be called."
            )

        # tokenize the text
        x = [self.tokenizer.tokenize(x_part) for x_part in x]

        # truncate the token sequences
        x = self._truncate(x)

        # convert the tokens to input ids and input mask.
        tokens = [self.tokenizer.cls_token] + x[0] + [self.tokenizer.sep_token]
        for x_part in x[1:]:
            tokens += (
                [self.tokenizer.sep_token] + x_part + [self.tokenizer.sep_token]
            )

        padding = [0 for _ in range(self.max_sequence_length - len(tokens))]

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens) + padding
        input_mask = [1 for _ in range(len(tokens))] + padding

        return {"input_ids": input_ids, "input_mask": input_mask}, y

    def _truncate(self, x: Sequence[List[str]]) -> Sequence[List[str]]:
        """Return ``x`` truncated to ``self.max_sequence_length``.

        Parameters
        ----------
        x : Sequence[List[str]]
            The sequence of tokenized strings for the instance.

        Returns
        -------
        Sequence[List[str]]
            The truncated tokenized strings for the instance.
        """
        # first, figure out how many tokens to keep for each string
        n_stacks = len(x)
        active_stacks = [
            [stack, space_in_stack]
            for stack, space_in_stack in enumerate(map(len, x))
        ]
        space_remaining = self.max_sequence_length - 2 * n_stacks
        tokens_to_keeps = [0 for _ in range(n_stacks)]
        while space_remaining > 0 and len(active_stacks) > 0:
            increment = space_remaining // n_stacks
            remainder = space_remaining % n_stacks
            for i, (stack, space_in_stack) in enumerate(active_stacks):
                used_space = (
                    min(space_in_stack, increment + 1)
                    if i < remainder
                    else min(space_in_stack, increment)
                )

                tokens_to_keeps[stack] += used_space
                space_remaining -= used_space
                active_stacks[i][1] -= used_space

                if space_in_stack == 0:
                    active_stacks.pop(i)

        # truncate the sequences
        if isinstance(self.truncation_strategy, str):
            truncation_strategies = [self.truncation_strategy] * len(x)
        else:
            truncation_strategies = self.truncation_strategy

        x = [
            x_part[:tokens_to_keep]
            if strategy == "beginning"
            else x_part[-tokens_to_keep:]
            for tokens_to_keep, x_part, strategy in zip(
                tokens_to_keeps, x, truncation_strategies
            )
        ]

        return x
