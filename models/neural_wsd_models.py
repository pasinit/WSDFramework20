import itertools
from abc import ABC
from nlp_models.huggingface_wrappers import GenericHuggingfaceWrapper
from torch.nn import Module, Linear, Parameter
from typing import Dict, Iterator
import torch

from data.datasets import Vocabulary


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
    def __init__(self, model_name, device, hidden_size, output_size, vocabulary: Vocabulary, lemma2synsets: Dict, encoder_trainable=False,
                 classifier_trainable=True):
        encoder = TransformerWSDEncoder(model_name, device)
        classifier = FeedForwardClassifier(hidden_size, output_size, vocabulary, lemma2synsets, device)
        self.device = device
        super().__init__(encoder, classifier, device, encoder_trainable=encoder_trainable, classifier_trainable=classifier_trainable)

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