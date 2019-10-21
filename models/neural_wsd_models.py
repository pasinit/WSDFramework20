from abc import ABC
import torch
from nlp_models.bert_wrappers import BertWrapper
from torch.nn import Module, Linear
from transformers import BertModel, GPT2Model, OpenAIGPTModel, TransfoXLModel, XLNetModel, XLMModel, RobertaModel, \
    DistilBertModel


class WSDClassifier(ABC):
    def __init__(self, num_classes, class2synset, lemmapos2synsets, **kwargs):
        self.num_classes = num_classes
        self.lemmapos2synsets = lemmapos2synsets
        self.class2synset = class2synset

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
        lemmapos_synsets = list()
        for i in range(len(probabilities)):
            lemmapos_scores.append(list())
            for j in range(len(probabilities[i])):
                sum = torch.sum(probabilities[i][j])
                if sum == 0.0:  # then it is masked
                    lemmapos_scores[-1].append(list())
                    lemmapos_synsets[-1].append(list())
                lemmapos = input_lemmapos[i][j]
                synsets = self.lemmapos2synsets.get(lemmapos, None)
                if synsets is not None:
                    classes = [self.class2synset[s] for s in synsets]
                    classes_scores = probabilities[i][j][classes]
                    lemmapos_scores[-1].append(list(zip(synsets, classes_scores)))
                else:
                    lemmapos_scores[-1].append(list())
                    lemmapos_synsets[-1].append(list())
        return lemmapos_scores


class WSDEncoder(ABC, Module):
    def __init__(self, **kwargs):
        super().__init__()


class WSDModel(Module):
    def __init__(self, encoder: WSDEncoder, classifier: WSDClassifier):
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier

    def forward(self, *input, **kwargs):
        encoded_states = self.encoder(input, **kwargs)
        predictions = self.classifier(encoded_states)
        return predictions


class FeedForwardClassifier(WSDClassifier, Module):
    def __init__(self, num_classes, class2synset, lemmapos2synsets, input_dim, **kwargs):
        super().__init__(num_classes, class2synset, lemmapos2synsets, **kwargs)
        self.input_dium = input_dim
        self.classifier = Linear(input_dim, num_classes)

    def classify(self, hidden_states, input_lemmapos, **kwargs):
        probabilities = self.classfier(hidden_states)
        return self.get_lemmapos_synset_scores(probabilities, input_lemmapos)


class BERTWSDEncoder(WSDEncoder):
    def __init__(self, model_name, device, **kwargs):
        super().__init__(**kwargs)
        self.model = BertWrapper(model_name, device)

    def forward(self, *input, **kwargs):
        return self.model.word_forward(input)["hidden_states"]


class BERTWSDModel():
    def __init__(self, model_name, device):
        encoder = BertWrapper(model_name, device)
        classifier = FeedForwardClassifier(117660, 768, dict(), dict())
