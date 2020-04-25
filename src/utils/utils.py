from allennlp.data.token_indexers import PretrainedTransformerIndexer


def get_token_indexer(model_name):
    if model_name.lower() == "nhs":
        model_name = "bert-base-multilingual-cased"
    indexer = PretrainedTransformerIndexer(
        model_name=model_name,
    )
    return indexer, indexer._tokenizer.pad_token_id
