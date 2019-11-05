from typing import Dict

from lxml import etree

from src.data.dataset_utils import get_pos_from_key, get_simplified_pos


class Lemma2Synsets(dict):
    def __init__(self, path: str = None, data: Dict = None, separator="\t", key_transform=lambda x: x,
                 value_transform=lambda x: x):
        """
        :param path: path to lemma 2 synset map.
        """
        assert not path or data
        super().__init__()
        if data is not None:
            self.update(data)
        else:
            with open(path) as lines:
                for line in lines:
                    fields = line.strip().split(separator)
                    key = key_transform(fields[0])
                    synsets = self.get(key, list())
                    synsets.extend([value_transform(v) for v in fields[1:]])
                    self[key] = synsets

    @staticmethod
    def load_keys(keys_path):
        key2gold = dict()
        with open(keys_path) as lines:
            for line in lines:
                fields = line.strip().split(" ")
                key2gold[fields[0]] = fields[1]
        return key2gold

    @staticmethod
    def from_corpus_xml(corpus_path, gold_transformer=lambda v: v):
        key_path = corpus_path.replace("data.xml", "gold.key.txt")
        key2gold = Lemma2Synsets.load_keys(key_path)
        root = etree.parse(corpus_path).getroot()
        lemmapos2gold = dict()
        for instance in root.findall("./text/sentence/instance"):
            tokenid = instance.attrib["id"]
            lemmapos = instance.attrib["lemma"] + "#" + get_simplified_pos(instance.attrib["pos"])
            lemmapos2gold[lemmapos] = lemmapos2gold.get(lemmapos, set())
            lemmapos2gold[lemmapos].add(key2gold[tokenid])
        for lemmapos, golds in lemmapos2gold.items():
            lemmapos2gold[lemmapos] = set(filter(lambda x: x is not None, [gold_transformer(g) for g in golds]))
        return Lemma2Synsets(data=lemmapos2gold)

    @staticmethod
    def offsets_from_wn_sense_index():
        lemmapos2gold = dict()
        with open("/opt/WordNet-3.0/dict/index.sense") as lines:
            for line in lines:
                fields = line.strip().split(" ")
                key = fields[0]
                pos = get_pos_from_key(key)
                offset = fields[1] + pos
                lexeme = key.split("%")[0] + "#" + pos
                golds = lemmapos2gold.get(lexeme, set())
                golds.add(offset)
                lemmapos2gold[lexeme] = golds
        return Lemma2Synsets(data=lemmapos2gold)

    @staticmethod
    def from_bn_mapping():
        lemmapos2gold = dict()
        with open("resources/lexeme_to_synsets/lexeme_to_bnoffsets.en.txt") as lines:
            for line in lines:
                fields = line.strip().split("\t")
                lemmapos = fields[0]
                synset = fields[1]
                synsets = lemmapos2gold.get(lemmapos, set())
                synsets.add(synset)
                lemmapos2gold[lemmapos] = synsets
        return Lemma2Synsets(data=lemmapos2gold)

    @staticmethod
    def sensekey_from_wn_sense_index():
        lemmapos2gold = dict()
        with open("/opt/WordNet-3.0/dict/index.sense") as lines:
            for line in lines:
                fields = line.strip().split(" ")
                key = fields[0]
                pos = get_pos_from_key(key)
                lexeme = key.split("%")[0] + "#" + pos
                golds = lemmapos2gold.get(lexeme, set())
                golds.add(key)
                lemmapos2gold[lexeme] = golds
        return Lemma2Synsets(data=lemmapos2gold)
