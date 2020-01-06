from typing import Dict, List

import torch
from allennlp.common.util import pad_sequence_to_length
from allennlp.data import Vocabulary, Token
from allennlp.data.token_indexers import WordpieceIndexer
from allennlp.data.token_indexers.wordpiece_indexer import _get_token_type_ids
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
    def tokens_to_indices(
            self, tokens: List[Token], vocabulary: Vocabulary, index_name: str
    ) -> Dict[str, List[int]]:
        if not self._added_to_vocabulary:
            self._add_encoding_to_vocabulary(vocabulary)
            self._added_to_vocabulary = True

        # This lowercases tokens if necessary
        text = (
            token.text.lower()
            if self._do_lowercase and token.text not in self._never_lowercase
            else token.text
            for token in tokens
        )

        # Obtain a nested sequence of wordpieces, each represented by a list of wordpiece ids
        token_wordpiece_ids = [
            [self.vocab.get(wordpiece, self.tokeniser.unk_token_id) for wordpiece in self.wordpiece_tokenizer(token)]
            for token in text
        ]

        # Flattened list of wordpieces. In the end, the output of the model (e.g., BERT) should
        # have a sequence length equal to the length of this list. However, it will first be split into
        # chunks of length `self.max_pieces` so that they can be fit through the model. After packing
        # and passing through the model, it should be unpacked to represent the wordpieces in this list.
        flat_wordpiece_ids = [wordpiece for token in token_wordpiece_ids for wordpiece in token]

        # Similarly, we want to compute the token_type_ids from the flattened wordpiece ids before
        # we do the windowing; otherwise [SEP] tokens would get counted multiple times.
        flat_token_type_ids = _get_token_type_ids(flat_wordpiece_ids, self._separator_ids)

        # The code below will (possibly) pack the wordpiece sequence into multiple sub-sequences by using a sliding
        # window `window_length` that overlaps with previous windows according to the `stride`. Suppose we have
        # the following sentence: "I went to the store to buy some milk". Then a sliding window of length 4 and
        # stride of length 2 will split them up into:

        # "[I went to the] [to the store to] [store to buy some] [buy some milk [PAD]]".

        # This is to ensure that the model has context of as much of the sentence as possible to get accurate
        # embeddings. Finally, the sequences will be padded with any start/end piece ids, e.g.,

        # "[CLS] I went to the [SEP] [CLS] to the store to [SEP] ...".

        # The embedder should then be able to split this token sequence by the window length,
        # pass them through the model, and recombine them.

        # Specify the stride to be half of `self.max_pieces`, minus any additional start/end wordpieces
        window_length = self.max_pieces - len(self._start_piece_ids) - len(self._end_piece_ids)
        stride = window_length // 2

        # offsets[i] will give us the index into wordpiece_ids
        # for the wordpiece "corresponding to" the i-th input token.
        offsets = []

        # If we're using initial offsets, we want to start at offset = len(text_tokens)
        # so that the first offset is the index of the first wordpiece of tokens[0].
        # Otherwise, we want to start at len(text_tokens) - 1, so that the "previous"
        # offset is the last wordpiece of "tokens[-1]".
        offset = (
            len(self._start_piece_ids)
            if self.use_starting_offsets
            else len(self._start_piece_ids) - 1
        )

        # Count amount of wordpieces accumulated
        pieces_accumulated = 0
        for token in token_wordpiece_ids:
            # Truncate the sequence if specified, which depends on where the offsets are
            next_offset = 1 if self.use_starting_offsets else 0
            if (
                    self._truncate_long_sequences
                    and offset + len(token) - 1 >= window_length + next_offset
            ):
                break

            # For initial offsets, the current value of ``offset`` is the start of
            # the current wordpiece, so add it to ``offsets`` and then increment it.
            if self.use_starting_offsets:
                offsets.append(offset)
                offset += len(token)
            # For final offsets, the current value of ``offset`` is the end of
            # the previous wordpiece, so increment it and then add it to ``offsets``.
            else:
                offset += len(token)
                offsets.append(offset)

            pieces_accumulated += len(token)

        if len(flat_wordpiece_ids) <= window_length:
            # If all the wordpieces fit, then we don't need to do anything special
            wordpiece_windows = [self._add_start_and_end(flat_wordpiece_ids)]
            token_type_ids = self._extend(flat_token_type_ids)
        elif self._truncate_long_sequences:
            self._warn_about_truncation(tokens)
            wordpiece_windows = [self._add_start_and_end(flat_wordpiece_ids[:pieces_accumulated])]
            token_type_ids = self._extend(flat_token_type_ids[:pieces_accumulated])
        else:
            # Create a sliding window of wordpieces of length `max_pieces` that advances by `stride` steps and
            # add start/end wordpieces to each window
            # TODO: this currently does not respect word boundaries, so words may be cut in half between windows
            # However, this would increase complexity, as sequences would need to be padded/unpadded in the middle
            wordpiece_windows = [
                self._add_start_and_end(flat_wordpiece_ids[i: i + window_length])
                for i in range(0, len(flat_wordpiece_ids), stride)
            ]

            token_type_windows = [
                self._extend(flat_token_type_ids[i: i + window_length])
                for i in range(0, len(flat_token_type_ids), stride)
            ]

            # Check for overlap in the last window. Throw it away if it is redundant.
            last_window = wordpiece_windows[-1][1:]
            penultimate_window = wordpiece_windows[-2]
            if last_window == penultimate_window[-len(last_window):]:
                wordpiece_windows = wordpiece_windows[:-1]
                token_type_windows = token_type_windows[:-1]

            token_type_ids = [token_type for window in token_type_windows for token_type in window]

        # Flatten the wordpiece windows
        wordpiece_ids = [wordpiece for sequence in wordpiece_windows for wordpiece in sequence]

        # Our mask should correspond to the original tokens,
        # because calling util.get_text_field_mask on the
        # "wordpiece_id" tokens will produce the wrong shape.
        # However, because of the max_pieces constraint, we may
        # have truncated the wordpieces; accordingly, we want the mask
        # to correspond to the remaining tokens after truncation, which
        # is captured by the offsets.
        mask = [1 for _ in offsets]

        return {
            index_name: wordpiece_ids,
            f"{index_name}-offsets": offsets,
            f"{index_name}-type-ids": token_type_ids,
            "mask": mask,
        }

    @overrides
    def as_padded_tensor(
            self,
            tokens: Dict[str, List[int]],
            desired_num_tokens: Dict[str, int],
            padding_lengths: Dict[str, int],
    ) -> Dict[str, torch.Tensor]:
        return {

            key: torch.LongTensor(
                pad_sequence_to_length(val, desired_num_tokens[key], default_value=lambda: self.tokeniser.pad_token_id))
            for key, val in tokens.items()
        }


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
