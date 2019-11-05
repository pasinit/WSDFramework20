from abc import ABC, abstractmethod

import torch
from nlp_models.bert_wrappers import BertWrapper
from torch.nn import Module

class WSDModel(ABC):
    def __init__(self, name, tokeniser=None):
        self.name = name
        self.tokeniser = tokeniser

    @abstractmethod
    def call(self, words, lemmas=None, poss=None):
        pass

    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)

    @abstractmethod
    def __disambiguate(self, *args):
        pass


class NeuralWSDModel(WSDModel):
    def __init__(self, name, tokeniser, device="cpu"):
        super().__init__(name, tokeniser)
        self.device = device

    def call(self, words, lemmas=None, poss=None):
        ids = torch.LongTensor(words).device
        return self.__disambiguate(ids)


class KBWSDModel(WSDModel):
    def __init__(self, name):
        super().__init__(name)

    def call(self, words, lemmas=None, poss=None):
        assert lemmas is not None and poss is not None
        disambiguations = list()
        for i_words, i_lemmas, i_poss in zip(words, lemmas, poss):
            i_disambiguations = self.__disambiguate(i_words, i_lemmas, i_poss)
            disambiguations.append(i_disambiguations)
        return disambiguations
