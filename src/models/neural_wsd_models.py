import itertools
from abc import ABC

from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import PretrainedBertEmbedder
from allennlp.training.metrics import CategoricalAccuracy, Metric
from allennlp_mods.callbacks import OutputWriter
from nlp_models.huggingface_wrappers import GenericHuggingfaceWrapper
from torch import nn
from torch.nn import Module, Linear, Parameter
from typing import Dict, Iterator, Any, Callable, List
import torch
import numpy as np
from src.data.data_structures import Lemma2Synsets
from src.data.datasets import Vocabulary, LabelVocabulary


class WSDClassifier(ABC):
    def __init__(self, num_classes, class2synset, lemmapos2synsets, device, **kwargs):
        self.num_classes = num_classes
        self.lemmapos2synsets = lemmapos2synsets
        self.class2synset = class2synset
        self.device = device

    def __call__(self, hidden_states, input_lemmapos, **kwargs):
        return self.classify(hidden_states, input_lemmapos, **kwargs)

    def classify(self, hidden_states, input_lemmapos, **kwargs):
        pass

    def get_lemmapos_synset_scores(self, probabilities, input_lemmapos):
        """
        :param probabilities: a tensor (batch x max_len x num_classes)
        :param input_lemmapos: a list that is parallel to the tensor probabilities
        :return: a list parallel to probabilities where in position i,j
         contains a list of pairs containing all the synsets corresponding to input_lemmapos[i,j] following the order in
         self.lemmapos2synsets together with their scores.
        """
        lemmapos_scores = list()
        mask = list()
        for i in range(len(input_lemmapos)):
            lemmapos_scores.append(list())
            mask.append(list())
            for j in range(len(input_lemmapos[i])):
                # sum = torch.sum(probabilities[i][j])
                # if sum == 0.0:  # then it is masked
                #     lemmapos_scores[-1].append(list())
                #     lemmapos_synsets[-1].append(list())
                lemmapos = input_lemmapos[i][j]
                synsets = self.lemmapos2synsets.get(lemmapos, None)
                zero_mask = torch.zeros_like(probabilities[0][0])
                if synsets is not None:
                    classes = [self.class2synset[s] for s in synsets if s in self.class2synset.stoi]
                    classes_scores = probabilities[i][j][classes]
                    zero_mask[classes] = 1.0
                    mask[-1].append(zero_mask)
                    lemmapos_scores[-1].append(list(zip(synsets, [s.item() for s in classes_scores])))

                else:
                    lemmapos_scores[-1].append(list())
                    mask[-1].append(zero_mask)
        return probabilities, lemmapos_scores, mask


class WSDEncoder(ABC, Module):
    def __init__(self, **kwargs):
        super().__init__()


class WSDModel(Module):
    def __init__(self, encoder: WSDEncoder, classifier: WSDClassifier, device, encoder_trainable=False,
                 classifier_trainable=True):
        super().__init__()
        self.encoder: WSDEncoder = encoder.to(device)
        self.classifier: WSDClassifier = classifier
        self.add_module("encoder", self.encoder)
        self.encoder_trainable = encoder_trainable
        self.classifier_trainable = classifier_trainable
        if isinstance(self.classifier, Module):
            self.add_module("classifier", self.classifier)

    def forward(self, input, input_lemmapos, **kwargs):
        if not self.encoder_trainable:
            with torch.no_grad():
                encoded_states = self.encoder(input, **kwargs)
        else:
            encoded_states = self.encoder(input, **kwargs)
        if not self.classifier_trainable:
            with torch.no_grad():
                predictions = self.classifier(encoded_states, input_lemmapos)
        else:
            predictions = self.classifier(encoded_states, input_lemmapos)
        return predictions

    def set_encoder_trainable(self):
        self.encoder.train()

    def set_classifier_trainable(self):
        self.classifier.train()

    def freeze_encoder_weights(self):
        self.encoder.eval()

    def freeze_classifier_weights(self):
        self.classifier.eval()

    def predict(self, input, input_lemmapos, **kwargs):
        with torch.no_grad():
            self.forward(input, input_lemmapos, **kwargs)


class FeedForwardClassifier(WSDClassifier, Module):
    def __init__(self, input_dim, num_classes, vocabulary: Vocabulary, lemmapos2synsets, device, **kwargs):
        WSDClassifier.__init__(self, num_classes, vocabulary, lemmapos2synsets, device, **kwargs)
        Module.__init__(self)
        self.input_dium = input_dim
        self.classifier = Linear(input_dim, num_classes).to(device)

    def classify(self, hidden_states, input_lemmapos, **kwargs):
        probabilities = self.classifier(hidden_states)
        return self.get_lemmapos_synset_scores(probabilities, input_lemmapos)


class TransformerWSDEncoder(WSDEncoder):
    def __init__(self, model_name, device, **kwargs):
        super().__init__(**kwargs)
        self.model: GenericHuggingfaceWrapper = GenericHuggingfaceWrapper(model_name, device).eval()
        self.add_module("transformer", self.model)

    def forward(self, input, **kwargs):
        out, _ = self.model.sentences_forward(input)
        return out["hidden_states"]


class TransformerFFWSDModel(WSDModel):
    def __init__(self, model_name, device, hidden_size, output_size, vocabulary: Vocabulary, lemma2synsets: Dict,
                 encoder_trainable=False,
                 classifier_trainable=True):
        encoder = TransformerWSDEncoder(model_name, device)
        classifier = FeedForwardClassifier(hidden_size, output_size, vocabulary, lemma2synsets, device)
        self.device = device
        super().__init__(encoder, classifier, device, encoder_trainable=encoder_trainable,
                         classifier_trainable=classifier_trainable)

    def parameters(self, recurse: bool = ...) -> Iterator[Parameter]:
        enc_params = None
        clas_params = None
        if self.encoder_trainable:
            enc_params = self.encoder.parameters()
        if self.classifier_trainable:
            clas_params = self.classifier.parameters()
        if enc_params and clas_params:
            return itertools.chain(enc_params, clas_params)
        if enc_params:
            return enc_params
        if clas_params:
            return clas_params
        return iter(())


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
            self.answers_mfs += 1
            if p == self.unk_id:
                p = self.mfs_vocab.get(lp, p)
            else:
                p = self.label_vocab.get_string(p)
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
                             "answers_mfs": self.answers_mfs, "tot_mfs":self.tot_mfs})

        if reset:
            self.tot = 0.0
            self.tot_mfs = 0.0
            self.answers = 0.0
            self.correct = 0.0
            self.correct_mfs = 0.0
            self.answers_mfs = 0.0
        return ret_dict


class AllenWSDModel(Model):
    def __init__(self, lemmapos2classes: Lemma2Synsets, word_embeddings: TextFieldEmbedder, out_sz, label_vocab,
                 vocab=None, merge_fun: Callable = lambda emb: torch.mean(emb, 0), mfs_vocab: Dict[str, str] = None,
                 return_full_output=False):
        vocab = Vocabulary() if vocab is None else vocab
        super().__init__(vocab)
        self.word_embeddings = word_embeddings
        self.word_embeddings.eval()
        self.projection = nn.Linear(self.word_embeddings.get_output_dim(), out_sz)
        self.loss = nn.CrossEntropyLoss()
        self.merge_fun = merge_fun
        self.label_vocab = label_vocab
        self.return_full_output = return_full_output
        self.lemma2classes = lemmapos2classes
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
                if batch_offset[o_j] == 0:
                    break
                end_offset = batch_offset[o_j] + 1
                embeddings_to_merge = batch_embeddings[start_offset:end_offset]
                start_offset = batch_offset[o_j] + 1
                token_level_emb = self.merge_fun(embeddings_to_merge)
                token_level_embeddings[batch_i, o_j] = token_level_emb
        return token_level_embeddings

    def get_embeddings(self, tokens, token_ids, mask):
        # token_ids = [t for i in range(len(token_ids)) for t in token_ids[i] if t is not None]
        # if self.vector_cache is not None and len(self.vector_cache) > 0 and self.training and all(
        #         x in self.vector_cache for x in token_ids):
        #     embeddings = torch.stack([self.vector_cache[x].to(self.projection.weight.device) for x in token_ids])
        # else:
        with torch.no_grad():
            embeddings = self.word_embeddings(tokens)
        embeddings = self.get_token_level_embeddings(embeddings, tokens["tokens-offsets"])
        retrieved_embedding_mask = mask != 0
        masked_embeddings = embeddings[retrieved_embedding_mask]
        # if token_ids is not None and self.training:
        #     self.vector_cache.update(dict(zip(token_ids, masked_embeddings)))
        #     assert all(torch.cat([self.vector_cache[x] for x in token_ids], 0) == masked_embeddings)
        embeddings = masked_embeddings

        return embeddings, retrieved_embedding_mask

    def forward(self, tokens: Dict[str, torch.Tensor], sentence_ids: List[str],
                ids: Any, words, lemmapos, label_ids: torch.Tensor,
                possible_labels, labeled_token_indices, labeled_lemmapos, labels) -> torch.Tensor:
        mask = (label_ids != self.label_vocab["<pad>"]).float().to(tokens["tokens"].device)
        embeddings, retrieved_embedding_mask = self.get_embeddings(tokens, ids, mask)
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

    def add_mfs(self, lemmapos, predictions):
        for i, lp, p in enumerate(zip(lemmapos, predictions)):
            if p == self.label_vocab.get_idx("<unk>"):
                mfs = self.mfs_dictionary.get(lemmapos, self.label_vocab.get_idx("<unk>"))
                predictions[i] = mfs

    def get_predictions(self, labeled_logits):
        predictions = list()
        for ll in labeled_logits:
            mask = (ll != 0).float()
            ll = torch.exp(ll) * mask
            predictions.append(torch.argmax(ll, -1))
        return torch.stack(predictions)

    @staticmethod
    def get_bert_based_wsd_model(bert_name, out_size, lemma2synsets: Lemma2Synsets, device, label_vocab,
                                 mfs_dictionary=None,
                                 vocab=None,
                                 return_full_output=False,
                                 cache_vectors=True):

        vocab = Vocabulary() if vocab is None else vocab
        bert_embedder = PretrainedBertEmbedder(
            pretrained_model=bert_name,
            top_layer_only=True,  # conserve memory
        )
        word_embeddings: TextFieldEmbedder = BasicTextFieldEmbedder({"tokens": bert_embedder},
                                                                    # we'll be ignoring masks so we'll need to set this to True
                                                                    allow_unmatched_keys=True)
        str_device = "cuda:{}".format(device) if device >= 0 else "cpu"
        word_embeddings.to(str_device)
        model = AllenWSDModel(lemma2synsets, word_embeddings, out_size, label_vocab, vocab, mfs_vocab=mfs_dictionary,
                              return_full_output=return_full_output)
        model.to(str_device)
        return model
