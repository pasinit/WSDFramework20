from typing import Dict, Any, Callable, List

import torch
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import PretrainedTransformerEmbedder
from allennlp.training.metrics import Metric
from allennlp_mods.callbacks import OutputWriter
from data_io.data_utils import Lemma2Synsets
from data_io.datasets import LabelVocabulary
from torch import nn

from src.data.datasets import Vocabulary


class WSDOutputWriter(OutputWriter):
    def __init__(self, output_file, labeldict):
        super().__init__(output_file, labeldict)

    def write(self, outs):
        predictions = outs["predictions"].flatten().tolist()
        golds = [x for y in outs["str_labels"] for x in y]
        ids = [x for y in outs["ids"] for x in y]
        assert len(predictions) == len(golds)
        for i in range(len(predictions)):  # p, l in zip(predictions, labels):
            p, l = predictions[i], "\t".join(golds[i])
            id = ids[i]
            out_str = (id if ids is not None else "") + "\t" + self.labeldict[p] + "\t" + l + "\n"
            self.writer.write(out_str)
        self.writer.flush()


class WSDF1(Metric):
    def __init__(self, label_vocab: LabelVocabulary, use_mfs: bool = False, mfs_vocab: Dict[str, int] = None):
        assert not use_mfs or mfs_vocab is not None
        self.correct = 0.0
        self.correct_mfs = 0.0
        self.tot = 0.0
        self.tot_mfs = 0.0
        self.answers = 0.0
        self.answers_mfs = 0.0
        self.mfs_vocab = mfs_vocab
        self.use_mfs = use_mfs
        self.unk_id = label_vocab.get_idx("<unk>") if "<unk>" in label_vocab.stoi else label_vocab.get_idx("<pad>")
        assert self.unk_id is not None
        self.label_vocab = label_vocab

    def compute_metrics_no_mfs(self, predictions, labels):
        for p, l in zip(predictions, labels):
            self.tot += 1
            if p == self.unk_id:
                continue
            self.answers += 1
            if self.label_vocab.get_string(p) in l:
                self.correct += 1

    def compute_metrics_mfs(self, lemmapos, predictions, labels):
        for lp, p, l in zip(lemmapos, predictions, labels):
            self.tot_mfs += 1

            if p == self.unk_id:
                p = self.mfs_vocab.get(lp, p)
            else:
                p = self.label_vocab.get_string(p)
            if p != self.unk_id:
                self.answers_mfs += 1
            if p in l:
                self.correct_mfs += 1

    def __call__(self, lemmapos, predictions, gold_labels, mask=None, ids=None):
        """
        :param predictions:
        :param gold_labels: assumes this is a List[List[Set[str]]] containing for each batch a list of Set each
        representing the possible gold labels for each token. This is parallel to predictions
        :param mask:
        :return:
        """
        assert len(predictions) == len(gold_labels)
        # p, r, f1 = get_metrics_no_mfs(predictions, gold_labels)
        # p_mfs, r_mfs, f1_mfs = get_matrics_mfs(lemmapos, predictions, gold_labels)
        self.compute_metrics_no_mfs(predictions, gold_labels)
        if self.mfs_vocab is not None:
            self.compute_metrics_mfs(lemmapos, predictions, gold_labels)

    def get_metric(self, reset: bool):
        if self.answers == 0:
            return {}
        precision = self.correct / self.answers
        recall = self.correct / self.tot
        f1 = 2 * (precision * recall) / ((precision + recall) if (precision + recall) > 0 else 1)
        ret_dict = {"precision": precision, "recall": recall, "f1": f1, "correct": self.correct,
                    "answers": self.answers,
                    "total": self.tot}
        if self.mfs_vocab is not None:
            precision_mfs = self.correct_mfs / self.answers_mfs
            recall_mfs = self.correct_mfs / self.tot_mfs
            f1_mfs = 2 * (precision_mfs * recall_mfs) / (
                (precision_mfs + recall_mfs) if (precision_mfs + recall_mfs) > 0 else 1)
            ret_dict.update({"p_mfs": precision_mfs,
                             "recall_mfs": recall_mfs,
                             "f1_mfs": f1_mfs, "correct_mfs": self.correct_mfs,
                             "answers_mfs": self.answers_mfs, "tot_mfs": self.tot_mfs})

        if reset:
            self.tot = 0.0
            self.tot_mfs = 0.0
            self.answers = 0.0
            self.correct = 0.0
            self.correct_mfs = 0.0
            self.answers_mfs = 0.0
        return ret_dict


@Model.register("textencoder_ff_wsd_classifier")
class AllenWSDModel(Model):
    def __init__(self, lemmapos2classes: Lemma2Synsets, word_embeddings: TextFieldEmbedder, out_sz, label_vocab,
                 vocab=None, merge_fun: Callable = lambda emb: torch.mean(emb, 0), mfs_vocab: Dict[str, str] = None,
                 return_full_output=False, finetune_embedder=False, cache_instances=False, pad_id=0):
        vocab = Vocabulary() if vocab is None else vocab
        super().__init__(vocab)
        self.finetune_embedder = finetune_embedder
        self.word_embeddings = word_embeddings
        if not finetune_embedder:
            self.word_embeddings.eval()
        else:
            self.word_embeddings.train()
        self.projection = nn.Linear(self.word_embeddings.get_output_dim(), out_sz)
        self.loss = nn.CrossEntropyLoss()
        self.merge_fun = merge_fun
        self.pad_id = pad_id
        self.label_vocab = label_vocab
        self.return_full_output = return_full_output
        self.lemma2classes = lemmapos2classes
        self.cache_instances = cache_instances
        self.cache = dict()
        self.accuracy = WSDF1(label_vocab, mfs_vocab is not None, mfs_vocab)

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return self.accuracy.get_metric(reset)

    def get_token_level_embeddings(self, embeddings, offsets):
        """
        :param embeddings:
        :param offsets: list of offsets where each original token **ends** in the list of subtokens.
        :return:
        """
        token_level_embeddings = torch.zeros(offsets.shape + embeddings.shape[-1:]).to(embeddings.device)
        for batch_i in range(len(offsets)):
            batch_offset = offsets[batch_i]
            batch_embeddings = embeddings[batch_i]
            start_offset = 1
            for o_j in range(len(batch_offset)):
                if batch_offset[o_j] == self.pad_id:
                    break
                end_offset = batch_offset[o_j] + 1
                embeddings_to_merge = batch_embeddings[start_offset:end_offset]
                start_offset = batch_offset[o_j] + 1
                token_level_emb = self.merge_fun(embeddings_to_merge)
                token_level_embeddings[batch_i, o_j] = token_level_emb
        return token_level_embeddings

    def get_embeddings_from_cache(self, tokens, mask, instance_ids):
        to_compute = [any(x not in self.cache for x in batch_instance_id) for batch_instance_id in instance_ids]
        indices_to_compute = [i for i in range(len(to_compute)) if to_compute[i]]

        if len(indices_to_compute) == 0:
            all_embeddings = list()
            for k, batch_instance_id in enumerate(instance_ids):
                embeddings = [self.cache[instance_id] for instance_id in batch_instance_id]
                all_embeddings.append(torch.stack(embeddings, 0).to(self.projection.weight.device))
            return torch.cat(all_embeddings, 0)
        tokens = {"tokens": tokens["tokens"][indices_to_compute],
                  "tokens-offsets": tokens["tokens-offsets"][indices_to_compute],
                  "tokens-type-ids": tokens["tokens-type-ids"][indices_to_compute],
                  "mask": tokens["mask"][indices_to_compute]}
        mask_to_compute = mask[indices_to_compute]
        embeddings = self.word_embeddings(tokens, **{"token_type_ids": tokens["tokens-type-ids"]})
        embeddings = self.get_token_level_embeddings(embeddings, tokens["tokens-offsets"])

        retrieved_embedding_mask = mask_to_compute != 0
        embeddings = embeddings[retrieved_embedding_mask]
        self.cache.update(dict(zip([x for y in instance_ids for x in y], embeddings.cpu())))
        return embeddings

    # def train(self, mode: bool = ...):
    #     self.projection.train()
    #     self.training = mode

    def get_embeddings(self, tokens, mask, instance_ids):
        retrieved_embedding_mask = mask != 0
        if not self.training:
            embeddings = self.word_embeddings(tokens)
        elif not self.finetune_embedder:
            if not self.cache_instances:
                with torch.no_grad():
                    embeddings = self.word_embeddings(tokens)
            else:
                return self.get_embeddings_from_cache(tokens, mask, instance_ids), retrieved_embedding_mask
        else:
            embeddings = self.word_embeddings(tokens)
        embeddings = self.get_token_level_embeddings(embeddings, tokens["tokens-offsets"])
        masked_embeddings = embeddings[retrieved_embedding_mask]
        embeddings = masked_embeddings

        return embeddings, retrieved_embedding_mask

    def forward(self, tokens: Dict[str, torch.Tensor], instance_ids: List[str],
                ids: Any, words, lemmapos, label_ids: torch.Tensor,
                possible_labels, labeled_token_indices, labeled_lemmapos, labels, cache_instance_ids) -> torch.Tensor:
        mask = (label_ids != self.label_vocab["<pad>"]).float().to(tokens["tokens"].device)

        embeddings, retrieved_embedding_mask = self.get_embeddings(tokens, mask, cache_instance_ids)
        labeled_logits = self.projection(embeddings)  # * mask.unsqueeze(-1)
        target_labels = label_ids[retrieved_embedding_mask]
        flatten_labels = [x for y in labels for x in y]
        possible_labels = [x for y in possible_labels for x in y]
        possible_classes_mask = torch.zeros_like(labeled_logits)  # .to(class_logits.device)
        for i, ith_lp in enumerate(possible_labels):
            possible_classes_mask[i][possible_labels[i]] = 1
            possible_classes_mask[:, 0] = 0
        labeled_logits = labeled_logits * possible_classes_mask
        predictions = None

        if not self.training:
            predictions = self.get_predictions(labeled_logits)
            self.accuracy([x for y in labeled_lemmapos for x in y], predictions.tolist(), flatten_labels)
        output = {"class_logits": labeled_logits, "all_logits": labeled_logits, "predictions": predictions,
                  "labels": labels, "all_labels": label_ids, "str_labels": labels,
                  "ids": [[x for x in i if x is not None] for i in ids], "loss":
                      self.loss(labeled_logits, target_labels)}
        if self.return_full_output:
            full_labeled_logits, full_predictions = self.reconstruct_full_output(retrieved_embedding_mask,
                                                                                 labeled_logits,
                                                                                 predictions)
            output.update({"full_labeled_logits": full_labeled_logits, "full_predictions": full_predictions})

        return output

    def reconstruct_full_output(self, retrieved_embedding_mask, labeled_logits, predictions):
        full_logits = torch.zeros(retrieved_embedding_mask.size(0), retrieved_embedding_mask.size(1),
                                  labeled_logits.size(-1)).to(predictions.device)
        full_predictions = torch.zeros(retrieved_embedding_mask.size(0), retrieved_embedding_mask.size(1)).to(
            predictions.device)
        index = 0
        for i, b_mask in enumerate(retrieved_embedding_mask):
            for j, elem in enumerate(b_mask):
                if elem.item():
                    full_logits[i][j] = labeled_logits[index]
                    full_predictions[i][j] = predictions[index]
                    index += 1

        return full_logits, full_predictions

    def get_predictions(self, labeled_logits):
        predictions = list()
        for ll in labeled_logits:
            mask = (ll != 0).float()
            ll = torch.exp(ll) * mask
            predictions.append(torch.argmax(ll, -1))
        return torch.stack(predictions)

    @staticmethod
    def get_transformer_based_wsd_model(model_name, out_size, lemma2synsets: Lemma2Synsets, device, label_vocab,
                                        pad_token_id=0,
                                        mfs_dictionary=None,
                                        vocab=None,
                                        return_full_output=False,
                                        eval=False, finetune_embedder=False, cache_vectors=False):
        vocab = Vocabulary() if vocab is None else vocab
        text_embedder = PretrainedTransformerEmbedder(pretrained_model=model_name,
                                                      top_layer_only=True,  # conserve memory
                                                      requires_grad=not eval and finetune_embedder,
                                                      pad_token_id=pad_token_id)
        # for param in text_embedder.parameters():
        #     param.requires_grad = finetune_embedder and not eval
        # if finetune_embedder:
        #     text_embedder.train()
        word_embeddings: TextFieldEmbedder = BasicTextFieldEmbedder({"tokens": text_embedder},
                                                                    # we'll be ignoring masks so we'll need to set this to True
                                                                    allow_unmatched_keys=True)
        str_device = "cuda:{}".format(device) if device >= 0 else "cpu"
        word_embeddings.to(str_device)
        model = AllenWSDModel(lemma2synsets, word_embeddings, out_size, label_vocab, vocab, mfs_vocab=mfs_dictionary,
                              return_full_output=return_full_output, cache_instances=cache_vectors, pad_id=pad_token_id,
                              finetune_embedder=finetune_embedder)
        model.to(str_device)
        return model

    # @staticmethod
    # def get_bert_based_wsd_model(bert_name, out_size, lemma2synsets: Lemma2Synsets, device, label_vocab,
    #                              mfs_dictionary=None,
    #                              vocab=None,
    #                              return_full_output=False,
    #                              eval=False, finetune_embedder=False, cache_vectors=False):
    #
    #     vocab = Vocabulary() if vocab is None else vocab
    #     bert_embedder = PretrainedBertEmbedder(
    #         pretrained_model=bert_name,
    #         top_layer_only=True,  # conserve memory
    #         requires_grad=not eval and finetune_embedder
    #     )
    #     word_embeddings: TextFieldEmbedder = BasicTextFieldEmbedder({"tokens": bert_embedder},
    #                                                                 # we'll be ignoring masks so we'll need to set this to True
    #                                                                 allow_unmatched_keys=True)
    #     str_device = "cuda:{}".format(device) if device >= 0 else "cpu"
    #     word_embeddings.to(str_device)
    #     model = AllenWSDModel(lemma2synsets, word_embeddings, out_size, label_vocab, vocab, mfs_vocab=mfs_dictionary,
    #                           return_full_output=return_full_output, cache_instances=cache_vectors)
    #     model.to(str_device)
    #     return model
