from collections import OrderedDict, Counter

import torch
from torch.utils.data.dataset import Dataset
import xml.etree.ElementTree as ET

from torchtext.vocab import Vocab


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


class WSDXmlInMemoryDataset(Dataset):
    def __init__(self, xml_path, key_file, vocabulary: Vocabulary, key2wnid_path=None, key2bnid_path=None):
        self.vocabulary = vocabulary
        key2gold = self.load_key_file(key_file, key2wnid_path, key2bnid_path)
        self.iwltc = self.load_xml(xml_path, key2gold)  # id word lemma tag class
        self.len_ = len(self.iwltc)

    def __getitem__(self, item):
        return self.iwltc[item]

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
    vocabulary = Vocabulary.vocabulary_from_gold_key_file(
        "/home/tommaso/Documents/data/WSD_Evaluation_Framework/Training_Corpora/SemCor/semcor.gold.key.txt")
    dataset = WSDXmlInMemoryDataset(
        "/home/tommaso/Documents/data/WSD_Evaluation_Framework/Evaluation_Datasets/ALL/ALL.data.xml",
        "/home/tommaso/Documents/data/WSD_Evaluation_Framework/Evaluation_Datasets/ALL/ALL.gold.key.txt", vocabulary)
    # key2bnid_path="/home/tommaso/dev/PycharmProjects/wsd_raganato_onesec/resources/all_bn_wn_keys.txt")
    # key2wnid_path="/opt/WordNet-3.0/dict/index.sense")

    print(dataset[10])
