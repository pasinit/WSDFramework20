import random
import xml.etree.ElementTree as ET
from collections import OrderedDict, Counter

import numpy as np
from data_io.batchers import TextDataset, ResettableIterator, ResettableListIterator, get_batcher
from torch.utils.data.dataset import Dataset
from torchtext.vocab import Vocab
from transformers import PreTrainedTokenizer, AutoTokenizer
from typing import Iterable, Iterator
from utils.huggingface_utils import encode_word_pieces


def get_pos_from_key(key):
    """
    assumes key is in the wordnet key format, i.e., 's_gravenhage%1:15:00:
    :param key: wordnet key
    :return: pos tag corresponding to the key
    """
    numpos = key.split("%")[-1]
    if numpos == "1":
        return "n"
    elif numpos == "2":
        return "v"
    elif numpos == "3":
        return "a"
    else:
        return "r"


def get_simplified_pos(long_pos):
    long_pos = long_pos.lower()
    if long_pos.startswith("n") or long_pos.startswith("propn"):
        return "n"
    elif long_pos.startswith("adj"):
        return "a"
    elif long_pos.startswith("adv"):
        return "r"
    elif long_pos.startswith("v"):
        return "v"
    return "o"


def load_wn_key2id_map(path):
    """
    assume the path points to a file in the same format of index.sense in WordNet dict/ subdirectory
    :param path: path to the file
    :return: dictionary from key to wordnet offsets
    """
    key2id = dict()
    with open(path) as lines:
        for line in lines:
            fields = line.strip().split(" ")
            key = fields[0]
            pos = get_pos_from_key(key)
            key2id[key] = ("wn:%08d" % int(fields[1])) + pos
    return key2id


def load_bn_key2id_map(path):
    """
    assumes the path points to a file with the following format:
    bnid\twn_key1\twn_key2\t...
    :param path:
    :return: a dictionary from wordnet key to bnid
    """
    key2bn = dict()
    with open(path) as lines:
        for line in lines:
            fields = line.strip().split("\t")
            bn = fields[0]
            for k in fields[1:]:
                key2bn[k] = bn
    return key2bn


class Vocabulary(Vocab):
    def __init__(self, counter, **kwargs):
        super().__init__(counter, **kwargs)
        self.itos = OrderedDict()
        for s, i in self.stoi.items():
            self.itos[i] = s

    def get_string(self, idx):
        return self.itos.get(idx, None)

    def get_idx(self, token):
        return self[token]

    @classmethod
    def vocabulary_from_gold_key_file(cls, gold_key, key2wnid_path=None, key2bnid_path=None):
        key2id = None
        if key2bnid_path:
            key2id = load_bn_key2id_map(key2bnid_path)
        elif key2wnid_path:
            key2id = load_wn_key2id_map(key2wnid_path)
        labels = Counter()
        with open(gold_key) as lines:
            for line in lines:
                fields = line.strip().split(" ")
                golds = fields[1:]
                if key2id is not None:
                    golds = [key2id[g] for g in golds]
                labels.update(golds)
        return Vocabulary(labels)


class WSDXmlInMemoryDataset(Iterator):
    def __init__(self, xml_path, key_file, vocabulary: Vocabulary, device, batch_size, key2wnid_path=None,
                 key2bnid_path=None, shuffle=True):

        self.vocabulary = vocabulary
        self.batch_size = batch_size
        self.device = device
        key2gold = self.load_key_file(key_file, key2wnid_path, key2bnid_path)
        self.iwltc = self.load_xml(xml_path, key2gold)  # id word lemma tag class
        self.all_sentences = list()
        for sentence in self.iwltc:
            self.all_sentences.append([tok[1] for tok in sentence])
        self.len_ = len(self.iwltc)
        # iterator = ResettableListIterator(all_sentences, shuffle=shuffle)
        # super().__init__(iterator, model_name, tokeniser, token_limit, device, batch_size,
        #                  max_sentences_in_memory=self.len_)
        # self.fill_buffer()
        self.indices = list(range(len(self.iwltc)))
        self.shuffle = shuffle
        if shuffle:
            random.shuffle(self.indices)

    def __next__(self):
        if len(self.indices) == 0:
            self.indices = list(range(len(self.iwltc)))
            if self.shuffle:
                random.shuffle(self.indices)
            raise StopIteration()

        words = list()
        ids = list()
        lemmas = list()
        poss = list()
        gold = list()
        for _ in range(self.batch_size):
            if len(self.indices) == 0:
                return words, ids, lemmas, poss, gold
            i = self.indices.pop()
            ids.append([x[0] for x in self.iwltc[i]])
            lemmas.append([x[2] for x in self.iwltc[i]])
            poss.append([x[3] for x in self.iwltc[i]])
            gold.append([x[4] for x in self.iwltc[i]])
            words.append(self.all_sentences[i])
        return words, ids, lemmas, poss, gold

    def __len__(self):
        return self.len_

    def load_key_file(self, f, key2wnid_path, key2bnid_path):
        key2id = None
        if key2wnid_path is not None:
            key2id = load_wn_key2id_map(key2wnid_path)
        elif key2bnid_path is not None:
            key2id = load_bn_key2id_map(key2bnid_path)
        key2gold = dict()
        with open(f) as lines:
            for line in lines:
                fields = line.strip().split(" ")
                key, *gold = fields
                if key2id:
                    gold = [key2id[k] for k in gold]
                classes = [self.vocabulary.get_idx(g) for g in gold]
                key2gold[key] = classes
        return key2gold

    def load_xml(self, xml_path, key2gold):
        root = ET.parse(xml_path).getroot()
        dataset = list()
        for sentence in root.findall("./text/sentence"):
            words = list()
            lemmas = list()
            poss = list()
            ids = list()
            labels = list()
            for elem in sentence:
                words.append(elem.text)
                lemmas.append(elem.attrib["lemma"])
                poss.append(elem.attrib["pos"])
                if elem.tag == "wf":
                    ids.append(None)
                    labels.append(None)
                else:
                    ids.append(elem.attrib["id"])
                    labels.append(key2gold[elem.attrib["id"]])
            dataset.append(list(zip(ids, words, lemmas, poss, labels)))
        return dataset


if __name__ == "__main__":
    device = "cuda"
    batch_size = 32
    vocabulary = Vocabulary.vocabulary_from_gold_key_file(
        "/home/tommaso/Documents/data/WSD_Evaluation_Framework/Training_Corpora/SemCor/semcor.gold.key.txt")
    dataset = WSDXmlInMemoryDataset(
        "/home/tommaso/Documents/data/WSD_Evaluation_Framework/Training_Corpora/SemCor/semcor.data.xml",
        "/home/tommaso/Documents/data/WSD_Evaluation_Framework/Training_Corpora/SemCor/semcor.gold.key.txt", vocabulary,
        device, batch_size)

    for batch in dataset:
        print(batch)
