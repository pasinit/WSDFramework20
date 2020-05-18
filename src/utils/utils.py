from allennlp.data import Vocabulary
from allennlp.data.token_indexers import PretrainedTransformerIndexer

from src.misc.wsdlogging import get_info_logger
from src.models import POSSIBLE_MODELS
from src.models.neural_wsd_models import AllenBatchNormWsdModel, AllenFFWsdModel

logger = get_info_logger(__name__)


def get_model(model_config, out_size, pad_token_id, label_pad_token_id, metric=None):
    wsd_model_name = model_config["wsd_model_name"]
    assert wsd_model_name in POSSIBLE_MODELS or logger.error(
        "WSD model not recognised: {}. Choose among {}".format(wsd_model_name, ",".join(POSSIBLE_MODELS)))
    if wsd_model_name == "ff_wsd_classifier":
        model_type = AllenFFWsdModel
    if wsd_model_name == "batchnorm_wsd_classifier":
        model_type = AllenBatchNormWsdModel

    model = model_type.get_transformer_based_wsd_model(**model_config,
                                                       out_size=out_size,
                                                       pad_id=pad_token_id,
                                                       label_pad_id=label_pad_token_id,
                                                       vocab=Vocabulary(),
                                                       model_path=model_config.get("model_path", None),
                                                       metric=metric)
    return model


def get_token_indexer(model_name):
    if model_name.lower() == "nhs":
        model_name = "bert-base-multilingual-cased"
    indexer = PretrainedTransformerIndexer(
        model_name=model_name,
    )
    return indexer, indexer._tokenizer.pad_token_id
