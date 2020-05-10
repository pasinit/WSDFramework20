from allennlp.data import Vocabulary
from allennlp.data.token_indexers import PretrainedTransformerIndexer

from src.misc.wsdlogging import get_info_logger
from src.models import POSSIBLE_MODELS
from src.models.neural_wsd_models import AllenBatchNormWsdModel, AllenFFWsdModel
logger = get_info_logger(__name__)


def get_model(cache_instances, device_int, encoder_name, finetune_embedder, label_vocab, layers_, lemma2synsets,
              mfs_dictionary, model_config, pad_token_id, wsd_model_name):
    assert wsd_model_name in POSSIBLE_MODELS or logger.error(
        "WSD model not recognised: {}. Choose among {}".format(wsd_model_name, ",".join(POSSIBLE_MODELS)))
    if wsd_model_name == "ff_wsd_classifier":
        model_type = AllenFFWsdModel
    if wsd_model_name == "batchnorm_wsd_classifier":
        model_type = AllenBatchNormWsdModel
    model = model_type.get_transformer_based_wsd_model(model_name=encoder_name,
                                                       out_size=len(label_vocab),
                                                       lemma2synsets=lemma2synsets,
                                                       device=device_int,
                                                       label_vocab=label_vocab,
                                                       pad_id=pad_token_id,
                                                       layers_=layers_,
                                                       vocab=Vocabulary(), mfs_dictionary=mfs_dictionary,
                                                       cache_vectors=cache_instances,
                                                       finetune_embedder=finetune_embedder,
                                                       model_path=model_config.get("model_path", None))
    return model
def get_token_indexer(model_name):
    if model_name.lower() == "nhs":
        model_name = "bert-base-multilingual-cased"
    indexer = PretrainedTransformerIndexer(
        model_name=model_name,
    )
    return indexer, indexer._tokenizer.pad_token_id
