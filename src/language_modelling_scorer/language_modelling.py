from typing import Dict, List

import torch
from allennlp.data import Vocabulary, Token
from allennlp.data.token_indexers import WordpieceIndexer
from allennlp.models import Model
from nlp_utils.huggingface_utils import get_needed_start_end_sentence_tokens
from overrides import overrides
from torch.nn import CrossEntropyLoss
from transformers import AutoModelWithLMHead, AutoTokenizer, WordpieceTokenizer, GPT2Model, add_start_docstrings, \
    GPT2PreTrainedModel, XLMWithLMHeadModel
import torch.nn as nn
from transformers.modeling_gpt2 import GPT2_START_DOCSTRING, GPT2_INPUTS_DOCSTRING


@add_start_docstrings("""The GPT2 Model transformer with a language modeling head on top
(linear layer with weights tied to the input embeddings). """, GPT2_START_DOCSTRING, GPT2_INPUTS_DOCSTRING)
class GPT2LMHeadModel(GPT2PreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Labels for language modeling.
            Note that the labels **are shifted** inside the model, i.e. you can set ``lm_labels = input_ids``
            Indices are selected in ``[-1, 0, ..., config.vocab_size]``
            All labels set to ``-1`` are ignored (masked), the loss is only
            computed for labels in ``[0, ..., config.vocab_size]``

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Language modeling loss.
        **prediction_scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, config.vocab_size)``
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        **past**:
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            that contains pre-computed hidden-states (key and values in the attention blocks).
            Can be used (see `past` input) to speed up sequential decoding.
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        import torch
        from transformers import GPT2Tokenizer, GPT2LMHeadModel

        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        model = GPT2LMHeadModel.from_pretrained('gpt2')

        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=input_ids)
        loss, logits = outputs[:2]

    """

    def __init__(self, config):
        super(GPT2LMHeadModel, self).__init__(config)
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.init_weights()
        self.tie_weights()

    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self._tie_or_clone_weights(self.lm_head,
                                   self.transformer.wte)

    def forward(self, input_ids, past=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                labels=None):
        transformer_outputs = self.transformer(input_ids,
                                               past=past,
                                               attention_mask=attention_mask,
                                               token_type_ids=token_type_ids,
                                               position_ids=position_ids,
                                               head_mask=head_mask)
        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)

        outputs = (lm_logits,) + transformer_outputs[1:]
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(ignore_index=-1, reduction="none")
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), lm_logits, presents, (all hidden_states), (attentions)


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
            truncate_long_sequences=truncate_long_sequences, **kwargs)

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
