import contextlib
import logging
import sys

import torch
from torch.nn import Module, CrossEntropyLoss, Linear, BatchNorm1d
from transformers import LxmertModel
from transformers.modeling_outputs import Seq2SeqLMOutput

logger = logging.getLogger(__name__)


def swish(x):
    return x * torch.sigmoid(x)


def get_empty_visual_features(batch_size):
    features = torch.zeros(batch_size, 1, 2048)
    pos = torch.zeros(batch_size, 1, 4) - 1
    return features, pos


class LxmertEncoderWrapper(Module):
    def __init__(self, model_name):
        super().__init__()
        self.encoder = LxmertModel.from_pretrained(model_name)

    def forward(self, input_ids=None,
                visual_feats=None,
                visual_pos=None,
                attention_mask=None,
                token_type_ids=None,
                visual_attention_mask=None,
                return_dict=True):
        if input_ids is None and visual_feats is None:
            logger.error("at least one among input_ids and visual_feats have to be set")
            sys.exit(1)
        dict_out = self.encoder(input_ids=input_ids,
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids,
                                visual_feats=visual_feats,
                                visual_pos=visual_pos,
                                visual_attention_mask=visual_attention_mask,
                                return_dict=return_dict)
        lang_encoded = dict_out.language_output

        dict_out["last_hidden_state"] = lang_encoded
        dict_out["extended_attention_mask"] = attention_mask
        return dict_out


class MultimodalWSDModel(Module):
    def __init__(self, output_vocab_size, multimodal_encoder="unc-nlp/lxmert-base-uncased",
                 finetune_encoder=False,
                 cache=True,
                 **kwargs):
        super().__init__()
        self.finetune_encoder = finetune_encoder
        self.encoder = LxmertEncoderWrapper(multimodal_encoder)
        embedding_size = self.encoder.encoder.config.hidden_size
        self.batchnorm = BatchNorm1d(embedding_size)
        self.linear = Linear(in_features=embedding_size, out_features=embedding_size)
        self.cache_vectors = not finetune_encoder and cache
        self.cache = dict()
        if not finetune_encoder:
            self.encoder.eval()
        self.classifier = Linear(in_features=embedding_size, out_features=output_vocab_size)

    def wsd_head(self, embeddings):
        if len(embeddings) > 1:
            embeddings = self.batchnorm(embeddings)

        embeddings = torch.dropout(embeddings, 0.5, self.training)
        embeddings = swish(self.linear(embeddings))
        return self.classifier(embeddings)  # mask.unsqueeze(-1)

    def update_cache(self, embeddings, token_ids):
        for e, ti in zip(embeddings, token_ids):
            self.cache[ti] = e.detach().cpu()

    def hit_in_cache(self, token_ids):
        embeddings = [self.cache.get(x, None) for x in token_ids]
        # print(embeddings)
        if all([x != None for x in embeddings]):
            return torch.stack(embeddings, 0).cuda()
        return None

    def get_embeddings(self, input_ids, attention_mask, token_type_ids, visual_feats, visual_pos,
                       visual_attention_mask, token_ids, return_dict, label_mask):
        encoded = self.hit_in_cache(token_ids)
        if encoded is not None:
            return encoded

        encoded = self.encoder(input_ids=input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids,
                               visual_feats=visual_feats,
                               visual_pos=visual_pos,
                               visual_attention_mask=visual_attention_mask,
                               return_dict=return_dict).last_hidden_state
        # print("NO CACHE, encoded shape", encoded.shape)
        encoded = self.get_relevant_embeddings(encoded, label_mask)
        # print("RELEVANT RETRIEVING ", encoded.shape)
        if self.cache_vectors:
            # print("CACHING VECTORS")
            self.update_cache(encoded, token_ids)
        return encoded

    def get_relevant_embeddings(self, embeddings, labels_mask):
        embeddings_flat = embeddings.contiguous().view(-1, embeddings.shape[-1])
        # print("IN RELEVANT EMBEDDINGS, embeddings_flat", embeddings_flat.shape)
        labels_mask = labels_mask.contiguous().view(-1).unsqueeze(-1)
        # print("LABEL_MASK", labels_mask.shape)
        embeddings_flat = embeddings_flat.masked_select(labels_mask).view(-1, embeddings.shape[-1])
        return embeddings_flat

    def forward(self, input_ids,
                text_attention_mask=None,
                token_type_ids=None,
                visual_feats=None,
                visual_pos=None,
                visual_attention_mask=None,
                labels=None,
                labels_mask=None,
                instance_ids=None,
                **kwargs):
        context = torch.no_grad() if not self.finetune_encoder or not self.training else contextlib.nullcontext()
        with context:
            encoded_input = self.get_embeddings(input_ids=input_ids,
                                                attention_mask=text_attention_mask,
                                                token_type_ids=token_type_ids,
                                                visual_feats=visual_feats,
                                                visual_pos=visual_pos,
                                                visual_attention_mask=visual_attention_mask,
                                                token_ids=instance_ids,
                                                return_dict=True,
                                                label_mask=labels_mask)

        logits = self.wsd_head(encoded_input)
        masked_lm_loss = None
        if labels is not None:
            # logits_for_loss = logits.contiguous().view(-1, logits.shape[-1])
            # labels_mask = labels_mask.contiguous().view(-1).unsqueeze(-1)
            # logits_for_loss = logits_for_loss.masked_select(labels_mask).view(labels.shape[0], -1)
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(logits, labels.view(-1))

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=logits,
        )

    def named_parameters(self, prefix: str = '', recurse: bool = True):
        all_params = list(super(MultimodalWSDModel, self).named_parameters(prefix, recurse))
        if self.finetune_encoder:
            return all_params
        return [(k, v) for k, v in all_params if not k.startswith("encoder.")]


if __name__ == "__main__":
    model = MultimodalWSDModel(117659)
    # print(len(model.named_parameters()))
    model = MultimodalWSDModel(117659, finetune_encoder=True)
    # print(len(model.named_parameters()))
