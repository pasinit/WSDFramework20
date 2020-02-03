from allennlp.data.token_indexers import PretrainedBertIndexer
from fairseq.models.bart import BARTModel

from src.models.core import PretrainedXLMIndexer, PretrainedRoBERTaIndexer


def get_token_indexer(model_name):
    if model_name.startswith("bert-"):
        return PretrainedBertIndexer(
            pretrained_model=model_name,
            do_lowercase=False,
            truncate_long_sequences=False
        ), 0
    if model_name.startswith("xlm-"):
        indexer = PretrainedXLMIndexer(
            pretrained_model=model_name,
            do_lowercase=False,
            truncate_long_sequences=False
        )
        return indexer, indexer.padding()
    if model_name.startswith("roberta-"):
        indexer = PretrainedRoBERTaIndexer(
            pretrained_model=model_name,
            do_lowercase=False,
            truncate_long_sequences=False
        )
        return indexer, indexer.padding()
    raise RuntimeError("Unknown model name: {}, cannot instanciate any indexer".format(model_name))