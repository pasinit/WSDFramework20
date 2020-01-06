import logging
import torch
from allennlp.data import Token, Vocabulary
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import TransformerEmbedder, PretrainedTransformerEmbedder
from transformers import XLMModel, XLMTokenizer

from src.models.core import PretrainedXLMIndexer

if __name__ == "__main__":
    tokenizer = XLMTokenizer.from_pretrained("xlm-mlm-en-2048")
    model = XLMModel.from_pretrained("xlm-mlm-en-2048")

    allen_indexer = PretrainedXLMIndexer("xlm-mlm-en-2048", truncate_long_sequences=False, do_lowercase=False)
    allen_model = PretrainedTransformerEmbedder("xlm-mlm-en-2048", tokenizer.pad_token_id, top_layer_only=True)
    allen_model = BasicTextFieldEmbedder({"tokens": allen_model},
                                                                # we'll be ignoring masks so we'll need to set this to True
                                                                allow_unmatched_keys=True)
    test_str = "this is a test"
    # indices = [tokenizer.encode(x) for x in test_str.split(" ")]
    indices = tokenizer.encode_plus(test_str, add_special_tokens=True)
    print(indices)
    out = model(torch.LongTensor(indices["input_ids"]).unsqueeze(0), token_type_ids=torch.LongTensor(indices["token_type_ids"]).unsqueeze(0))[0]
    print(out.shape)
    print(out)

    allen_indices = allen_indexer.tokens_to_indices([Token(x) for x in test_str.split(" ")], Vocabulary(), "tokens")
    print(allen_indices)

    allen_out = allen_model({k: torch.LongTensor(v).unsqueeze(0) for k,v in allen_indices.items()},
                            **{"token_type_ids":torch.LongTensor(allen_indices["tokens-type-ids"]).unsqueeze(0)})
        #(torch.LongTensor(allen_indices["tokens"]).unsqueeze(0), offsets=torch.LongTensor(allen_indices["tokens-offsets"]).unsqueeze(0),
         #       token_type_ids=torch.LongTensor(allen_indices["tokens-type-ids"]).unsqueeze(0))
    print(allen_out.shape)
    print(allen_out)
