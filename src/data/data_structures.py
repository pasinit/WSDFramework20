# from typing import Dict
#
# from data_io.data_utils import Lemma2Synsets
# from lxml import etree
# from src.data.dataset_utils import get_pos_from_key, get_simplified_pos
#
#
# # class Lemma2Synsets(dict):
# #     def __init__(self, path: str = None, data: Dict = None, separator="\t", key_transform=lambda x: x,
# #                  value_transform=lambda x: x):
# #         """
# #         :param path: path to lemma 2 synset map.
# #         """
# #         assert not path or data
# #         super().__init__()
# #         if data is not None:
# #             self.update(data)
# #         else:
# #             with open(path) as lines:
# #                 for line in lines:
# #                     fields = line.strip().split(separator)
# #                     key = key_transform(fields[0])
# #                     synsets = self.get(key, list())
# #                     synsets.extend([value_transform(v) for v in fields[1:]])
# #                     self[key] = synsets
# #
# #     @staticmethod
# #     def load_keys(keys_path):
# #         key2gold = dict()
# #         with open(keys_path) as lines:
# #             for line in lines:
# #                 fields = line.strip().split(" ")
# #                 key2gold[fields[0]] = fields[1]
# #         return key2gold
# #
# #     @staticmethod
# #     def from_corpus_xml(corpus_path, gold_transformer=lambda v: v):
# #         key_path = corpus_path.replace("data.xml", "gold.key.txt")
# #         key2gold = Lemma2Synsets.load_keys(key_path)
# #         # root = etree.parse(corpus_path).getroot()
# #         lemmapos2gold = dict()
# #         for _, instance in etree.iterparse(corpus_path, tag="instance", events=("start",)):
# #             tokenid = instance.attrib["id"]
# #             lemmapos = instance.attrib["lemma"].lower() + "#" + get_simplified_pos(instance.attrib["pos"])
# #             lemmapos2gold[lemmapos] = lemmapos2gold.get(lemmapos, set())
# #             lemmapos2gold[lemmapos].add(key2gold[tokenid].replace("%5", "%3"))
# #         for lemmapos, golds in lemmapos2gold.items():
# #             lemmapos2gold[lemmapos] = set(filter(lambda x: x is not None, [gold_transformer(g) for g in golds]))
# #         return Lemma2Synsets(data=lemmapos2gold)