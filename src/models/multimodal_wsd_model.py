from transformers import AutoModel
import contextlib
import logging
import sys

import torch
from torch import nn
from torch.nn import Module, CrossEntropyLoss, Linear, BatchNorm1d
from torch.nn.modules.activation import Sigmoid
from transformers.configuration_utils import PretrainedConfig
from transformers.configuration_xlnet import XLNetConfig
from transformers.modeling_bert import BertSelfAttention
from transformers.modeling_lxmert import LxmertAttention, LxmertCrossAttentionLayer
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers import BertModel

logger = logging.getLogger(__name__)


def swish(x):
    return x * torch.sigmoid(x)


def get_empty_visual_features(batch_size):
    features = torch.zeros(batch_size, 1, 2048)
    pos = torch.zeros(batch_size, 1, 4) - 1
    return features, pos


class MultimodalWSDModel(Module):
    def __init__(self, output_vocab_size, multimodal_encoder,
                 finetune_encoder=False,
                 cache=True,
                 **kwargs):
        super().__init__()
        self.finetune_encoder = finetune_encoder
        self.encoder = AutoModel.from_pretrained(multimodal_encoder)
        self.embedding_size = embedding_size = self.encoder.config.hidden_size
        self.batchnorm = BatchNorm1d(embedding_size)
        self.linear = Linear(in_features=embedding_size,
                             out_features=embedding_size)
        self.cache_vectors = not finetune_encoder and cache
        self.cache = dict()
        if not finetune_encoder:
            self.encoder.eval()
        self.classifier = Linear(
            in_features=embedding_size, out_features=output_vocab_size)

    def wsd_head(self, embeddings, **kwargs):
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
                               return_dict=return_dict).language_output
        # print("NO CACHE, encoded shape", encoded.shape)
        encoded = self.get_relevant_embeddings(encoded, label_mask)
        # print("RELEVANT RETRIEVING ", encoded.shape)
        if self.cache_vectors:
            # print("CACHING VECTORS")
            self.update_cache(encoded, token_ids)
        return encoded

    def get_relevant_embeddings(self, embeddings, labels_mask):
        embeddings_flat = embeddings.contiguous(
        ).view(-1, embeddings.shape[-1])
        # print("IN RELEVANT EMBEDDINGS, embeddings_flat", embeddings_flat.shape)
        labels_mask = labels_mask.contiguous().view(-1).unsqueeze(-1)
        # print("LABEL_MASK", labels_mask.shape)
        embeddings_flat = embeddings_flat.masked_select(
            labels_mask).view(-1, embeddings.shape[-1])
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
        context = torch.no_grad(
        ) if not self.finetune_encoder or not self.training else contextlib.nullcontext()
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

        logits = self.wsd_head(encoded_input, **kwargs)
        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(logits, labels.view(-1))

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=logits,
        )

    def named_parameters(self, prefix: str = '', recurse: bool = True):
        all_params = list(
            super(MultimodalWSDModel, self).named_parameters(prefix, recurse))
        if self.finetune_encoder:
            return all_params
        return [(k, v) for k, v in all_params if not k.startswith("encoder.")]


class BertFusionWSDModel(MultimodalWSDModel):
    def __init__(self, output_vocab_size, multimodal_encoder, finetune_encoder, cache,
                 img_feature_size: int, **kwargs):
        super().__init__(output_vocab_size, multimodal_encoder,
                         finetune_encoder=finetune_encoder, cache=cache, **kwargs)
        self.text_fusion_transformation = nn.Linear(
            self.embedding_size, self.embedding_size)
        self.image_fusion_transformation = nn.Linear(
            img_feature_size, self.embedding_size)
        self.sigmoid = Sigmoid()

    def fuse(self, text_embeddings, img_embeddings):
        """[summary]
        Args:
            text_embeddings (torch.Tensor): Text hidden states (batch, words, t-hdim) 
            img_embeddings (torch.Tensor): Image hidden states (batch, i-hdim)
        """
        img_embeddings = torch.mean(img_embeddings, 1)
        img_embeddings = self.image_fusion_transformation(
            img_embeddings).unsqueeze(1).repeat(
            1, text_embeddings.size(1), 1)  # (batch, words, hdim)
        text_embeddings = self.text_fusion_transformation(
            text_embeddings).view(*img_embeddings.shape[:2], -1)  # (batch, words, hdim)
        # (batch, words, hdim)
        gate = self.sigmoid(img_embeddings+text_embeddings)
        gated_img_embeddings = img_embeddings * gate
        fused_embeddings = text_embeddings + gated_img_embeddings
        return fused_embeddings

    def get_embeddings(self, input_ids, attention_mask, token_type_ids,
                       visual_feats,
                       visual_pos,
                       visual_attention_mask,
                       token_ids,
                       return_dict,
                       label_mask):
        encoded = self.hit_in_cache(token_ids)
        if encoded is not None:
            return encoded

        encoded = self.encoder(input_ids=input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids,
                               return_dict=True).last_hidden_state
        encoded = self.fuse(encoded, visual_feats)

        encoded = self.get_relevant_embeddings(encoded, label_mask)

        if self.cache_vectors:
            self.update_cache(encoded, token_ids)
        return encoded


class BertCrossAttentionWSDModel(BertFusionWSDModel):
    def __init__(self, output_vocab_size, multimodal_encoder, finetune_encoder, cache,
                 img_feature_size, **kwargs):
        super().__init__(output_vocab_size, multimodal_encoder,
                         finetune_encoder=finetune_encoder, cache=cache, img_feature_size=img_feature_size,
                         **kwargs)
        config = PretrainedConfig(num_attention_heads=1,
                                  hidden_size=self.embedding_size,
                                  attention_head_size=self.embedding_size,
                                  attention_probs_dropout_prob=0.1)
        self.cross_attention = LxmertAttention(config, ctx_dim=2048)

    def fuse(self, text_embeddings, img_embeddings):
        """[summary]
        Args:
            text_embeddings (torch.Tensor): Text hidden states (batch, words, t-hdim) 
            img_embeddings (torch.Tensor): Image hidden states (batch, num_feats, i-hdim)
        """
        img_embeddings, *_ = self.cross_attention(
            text_embeddings, img_embeddings)  # (batch, words, i-hdim)
        text_embeddings = self.text_fusion_transformation(
            text_embeddings)  # (batch, words, hdim)
        # (batch, words, hdim)
        # (batch, words, hdim)
        gate = self.sigmoid(img_embeddings+text_embeddings)
        gated_img_embeddings = img_embeddings * gate
        fused_embeddings = text_embeddings + gated_img_embeddings
        return fused_embeddings
