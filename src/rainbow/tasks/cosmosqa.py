"""The CosmosQA task."""

from fairseq.tasks import register_task

from . import base


@register_task("cosmosqa")
class CosmosQATask(base.MultipleChoiceTask):
    """The CosmosQA task."""

    def load_dataset(self, split, combine=False, **kwargs):
        raise NotImplementedError
