import torch
import torch.nn as nn
import transformers
import contextlib
from transformers.modeling_outputs import Seq2SeqLMOutput


def swish(x):
    return x * torch.sigmoid(x)


class Swish(nn.Module):
    def forward(self, x):
        return swish(x)


class TransformerWSDModel(torch.nn.Module):
    def __init__(self, encoder_name, out_vocab, finetune_encoder, cache=True) -> None:
        super().__init__()
        self.encoder = transformers.AutoModel.from_pretrained(encoder_name)
        embedding_size = self.encoder.encoder.config.hidden_size
        self.wsd_head = nn.Sequential(
            nn.BatchNorm1d(self.encoder.encoder.config.hidden_size),
            nn.Dropout(0.5),
            nn.Linear(embedding_size, embedding_size),
            Swish(),
            nn.Linear(embedding_size, out_vocab),
        )
        self.finetune_encoder = finetune_encoder
        self.cache_vectors = not finetune_encoder and cache
        self.cache = dict()
        if not self.finetune_encoder:
            self.encoder.eval()

    def update_cache(self, embeddings, token_ids):
        for e, ti in zip(embeddings, token_ids):
            self.cache[ti] = e.detach().cpu()

    def hit_in_cache(self, token_ids):
        embeddings = [self.cache.get(x, None) for x in token_ids]
        # print(embeddings)
        if all([x != None for x in embeddings]):
            return torch.stack(embeddings, 0).cuda()
        return None

    def get_encoder_inputs(self, **kwargs):
        names = self.encoder.forward.__code__.co_varnames
        new_args = dict()
        for k, v in kwargs.items():
            if k in names:
                new_args[k] = v
        return new_args

    def run_encoder(self, **kwargs):
        args = self.get_encoder_inputs(**kwargs)
        return self.encoder(**args)

    def get_embeddings(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        visual_feats,
        visual_pos,
        visual_attention_mask,
        token_ids,
        return_dict,
        label_mask,
    ):
        encoded = self.hit_in_cache(token_ids)
        if encoded is not None:
            return encoded

        encoded = self.run_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            visual_feats=visual_feats,
            visual_pos=visual_pos,
            visual_attention_mask=visual_attention_mask,
            return_dict=return_dict,
        ).last_hidden_state
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
        embeddings_flat = embeddings_flat.masked_select(labels_mask).view(
            -1, embeddings.shape[-1]
        )
        return embeddings_flat

    def forward(
        self,
        input_ids,
        text_attention_mask=None,
        token_type_ids=None,
        visual_feats=None,
        visual_pos=None,
        visual_attention_mask=None,
        labels=None,
        labels_mask=None,
        instance_ids=None,
        **kwargs
    ):
        context = (
            torch.no_grad()
            if not self.finetune_encoder or not self.training
            else contextlib.nullcontext()
        )
        with context:
            encoded_input = self.get_embeddings(
                input_ids=input_ids,
                attention_mask=text_attention_mask,
                token_type_ids=token_type_ids,
                visual_feats=visual_feats,
                visual_pos=visual_pos,
                visual_attention_mask=visual_attention_mask,
                token_ids=instance_ids,
                return_dict=True,
                label_mask=labels_mask,
            )

        logits = self.wsd_head(encoded_input)
        masked_lm_loss = None
        if labels is not None:
            # logits_for_loss = logits.contiguous().view(-1, logits.shape[-1])
            # labels_mask = labels_mask.contiguous().view(-1).unsqueeze(-1)
            # logits_for_loss = logits_for_loss.masked_select(labels_mask).view(labels.shape[0], -1)
            loss_fct = nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(logits, labels.view(-1))

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=logits,
        )
    # def named_parameters(self):
    #     if self.finetune_encoder:
    #         return super().named_parameters()
    #     return self.wsd_head.named_parameters()