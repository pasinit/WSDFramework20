from typing import Dict, List

import torch
from allennlp.data import Vocabulary, Token
from allennlp.data.token_indexers import WordpieceIndexer
from allennlp.models import Model
from nlp_utils.huggingface_utils import get_needed_start_end_sentence_tokens
from overrides import overrides
from transformers import AutoTokenizer

from src.language_modelling_scorer.gpt2_mod import GPT2LMHeadModel
from src.language_modelling_scorer.xlm_mod import XLMWithLMHeadModel


class AutoHuggingfaceIndexer(WordpieceIndexer):
    def __init__(self, pretrained_model, use_starting_offsets: bool = False,
                 do_lowercase: bool = True,
                 never_lowercase: List[str] = None,
                 max_pieces: int = 512,
                 truncate_long_sequences: bool = True,
                 **kwargs):
        self.tokeniser = tokeniser = AutoTokenizer.from_pretrained(pretrained_model)
        self.mask_token_id = self.tokeniser.mask_token_id
        wordpiece_tokenizer = tokeniser.tokenize
        start, end = get_needed_start_end_sentence_tokens(pretrained_model, tokeniser)
        self.model_name = pretrained_model
        # self.add_prefix_space = False
        # if "roberta" in self.model_name.lower() or "gpt2" in self.model_name.lower():
        #     self.add_prefix_space = True
        if start is None:
            start = ""
        if end is None:
            end = ""
        sep = tokeniser.sep_token if hasattr(tokeniser, "sep_token") else ""
        if sep is None:
            sep = end
        super().__init__(
            vocab=tokeniser.encoder,
            wordpiece_tokenizer=wordpiece_tokenizer,
            namespace=pretrained_model,
            use_starting_offsets=use_starting_offsets,
            max_pieces=max_pieces,
            do_lowercase=do_lowercase,
            never_lowercase=never_lowercase,
            start_tokens=[start],
            end_tokens=[end],
            separator_token=sep,
            truncate_long_sequences=truncate_long_sequences)

    @overrides
    def tokens_to_indices(self, tokens, vocabulary, index_name):
        return super().tokens_to_indices(tokens, vocabulary, index_name)




class AutoHuggingfaceLM(Model):
    def __init__(self, vocab: Vocabulary, model_name: str):
        super().__init__(vocab)
        self.model = GPT2LMHeadModel.from_pretrained(
            model_name) if model_name == "gpt2" else XLMWithLMHeadModel.from_pretrained(model_name)

    def forward(self, tokens, metadata) -> Dict[str, torch.Tensor]:
        segment_ids = tokens["tokens"]
        loss, scores, *_ = self.model(segment_ids, labels=segment_ids)
        return {"loss": loss, "scores": scores}


if __name__ == "__main__":
    indexer = AutoHuggingfaceIndexer(
        pretrained_model="gpt2",  # xlm-mlm-100-1280",
        do_lowercase=False,
        truncate_long_sequences=False,
        use_starting_offsets=True
    )
    test_str = "this is a very simple test to verify the indexer class"
    tokens = [Token(x) for x in test_str.split(" ")]
    indices = indexer.tokens_to_indices(tokens, Vocabulary(), "tokens")
    indices["tokens"] = torch.LongTensor(indices["tokens"]).unsqueeze(0)
    print(indices)
    model = AutoHuggingfaceLM(Vocabulary(), "xlm-mlm-100-1280")
    out = model(indices)
    print(out)
