from collections import Counter
from typing import Dict, Any, Tuple, Union, Callable

import torch
from allennlp.data.dataset_readers.dataset_reader import AllennlpDataset
from allennlp.data.token_indexers import PretrainedTransformerIndexer, PretrainedTransformerMismatchedIndexer
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp_training_callbacks.callbacks import OutputWriter
from data_io.data_utils import Lemma2Synsets
from data_io.datasets import LabelVocabulary, WSDDataset
import logging

from deprecated import deprecated

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

WORDNET_DICT_PATH = "/opt/WordNet-3.0/dict/index.sense"


def offsets_from_wn_sense_index():
    lemmapos2gold = dict()
    with open("/opt/WordNet-3.0/dict/index.sense") as lines:
        for line in lines:
            fields = line.strip().split(" ")
            key = fields[0]
            pos = get_pos_from_key(key)
            offset = "wn:" + fields[1] + pos
            lexeme = key.split("%")[0] + "#" + pos
            golds = lemmapos2gold.get(lexeme, set())
            golds.add(offset)
            lemmapos2gold[lexeme] = golds
    return Lemma2Synsets(data=lemmapos2gold)


def from_bn_mapping(langs=("en"), sense_inventory=None, **kwargs):
    reliable = True
    # if sense_inventory is not None and "bnoffsets" in sense_inventory:
    #     reliable = "bnoffsets_reliable" == sense_inventory
    lemmapos2gold = dict()
    for lang in langs:
        with open("resources/inventory/inventory.{}.withgold.txt".format(lang)) as lines:
            for line in lines:
                fields = line.strip().lower().split("\t")
                if len(fields) < 2:
                    continue
                lemmapos = fields[0]
                synsets = fields[1:]
                old_synsets = lemmapos2gold.get(lemmapos, set())
                old_synsets.update(synsets)
                lemmapos2gold[lemmapos] = old_synsets
    return Lemma2Synsets(data=lemmapos2gold)


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


def load_bn_offset2bnid_map(path):
    offset2bnid = dict()
    with open(path) as lines:
        for line in lines:
            fields = line.strip().split("\t")
            bnid = fields[0]
            for wnid in fields[1:]:
                offset2bnid[wnid] = bnid
    return offset2bnid


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


def vocabulary_from_gold_key_file(gold_key, key2wnid_path=None, key2bnid_path=None):
    key2id = None
    if key2bnid_path:
        key2id = load_bn_key2id_map(key2bnid_path)
    elif key2wnid_path:
        key2id = load_wn_key2id_map(key2wnid_path)
    labels = Counter()
    with open(gold_key) as lines:
        for line in lines:
            fields = line.strip().split(" ")
            golds = [x.replace("%5", "%3") for x in fields[1:]]
            if key2id is not None:
                golds = [key2id[g] for g in golds]
            labels.update(golds)
    return LabelVocabulary(labels, specials=["<pad>", "<unk>"])


def wnoffset_vocabulary():
    offsets = list()
    with open(WORDNET_DICT_PATH) as lines:
        for line in lines:
            fields = line.strip().split(" ")
            key = fields[0]
            pos = get_pos_from_key(key)
            offset = "wn:" + fields[1] + pos
            offsets.append(offset)
    return LabelVocabulary(Counter(sorted(offsets)), specials=["<pad>", "<unk>"])


def bnoffset_vocabulary():
    # with open("resources/vocabularies/bn_vocabulary.txt") as lines:
    #     bnoffsets = [line.strip() for line in lines]
    wn2bn = get_wnoffset2bnoffset()
    offsets = set()
    with open(WORDNET_DICT_PATH) as lines:
        for line in lines:
            fields = line.strip().split(" ")
            key = fields[0]
            pos = get_pos_from_key(key)
            offset = "wn:" + fields[1] + pos
            bnoffset = wn2bn[offset]
            offsets.update(bnoffset)
    return LabelVocabulary(Counter(sorted(offsets)), specials=["<pad>", "<unk>"])


def wn_sensekey_vocabulary():
    with open(WORDNET_DICT_PATH) as lines:
        keys = [line.strip().split(" ")[0].replace("%5", "%3") for line in lines]
    return LabelVocabulary(Counter(sorted(keys)), specials=["<pad>", "<unk>"])


def get_label_mapper(target_inventory, labels):
    label_types = set(
        ["wnoffsets" if l.startswith("wn:") else "bnoffsets" if l.startswith("bn:") else "sensekeys" for l in labels if
         l != "<pad>" and l != "<unk>"])
    if target_inventory in label_types:
        label_types.remove(target_inventory)
    if len(label_types) > 1:
        raise RuntimeError(
            "cannot handle the mapping from 2 or more label types ({}) to the target inventory {}".format(
                ",".join(label_types), target_inventory))
    if len(label_types) == 0:
        return None
    label_type = next(iter(label_types))
    if label_type == "wnoffsets":
        if target_inventory == "bnoffsets":
            return get_wnoffset2bnoffset()
        elif target_inventory == "sensekeys":
            return get_wnoffset2wnkeys()
        return None
    elif label_type == "sensekeys" is not None:
        if target_inventory == "bnoffsets":
            return get_wnkeys2bnoffset()
        elif target_inventory == "wnoffsets":
            return get_wnkeys2wnoffset()
        else:
            return None
    else:
        if target_inventory == "wnoffsets":
            return get_bnoffset2wnoffset()
        elif target_inventory == "sensekeys":
            return get_bnoffset2wnkeys()
        else:
            raise RuntimeError("Cannot infer label type from {}".format(label_type))


def get_mfs_vocab(mfs_file):
    if mfs_file is None:
        return None
    mfs = dict()
    with open(mfs_file) as lines:
        for line in lines:
            fields = line.strip().lower().split("\t")
            if len(fields) < 2:
                continue
            mfs[fields[0].lower()] = fields[1].replace("%5", "%3")
    return mfs


def get_dataset(model_name: str,
                paths: Union[str, list],
                lemma2synsets: Lemma2Synsets,
                mfs_file: str,
                label_mapper: Dict[str, str],
                label_vocab: LabelVocabulary) \
        -> Tuple[AllennlpDataset, Lemma2Synsets, Dict]:
    indexer = PretrainedTransformerMismatchedIndexer(model_name)
    dataset = WSDDataset(paths, lemma2synsets=lemma2synsets, label_mapper=label_mapper,
                         indexer=indexer, label_vocab=label_vocab)
    mfs_vocab = get_mfs_vocab(mfs_file)
    return dataset, lemma2synsets, mfs_vocab


@deprecated(reason="this method is not maintained anymore, it might brake things")
def get_dataset_with_labels_from_data(model_name, paths, label_mapper, langs, mfs_file=None, **kwargs):
    labels = set()
    lemma2synsetslist = list()
    for path in paths:
        with open(path.replace("data.xml", "gold.key.txt")) as lines:
            for line in lines:
                labels.update(line.strip().split(" ")[1:])
    labels = list([list(label_mapper.get(x, {x}))[0] if label_mapper is not None else x for x in labels])
    labels = sorted(labels)
    label_vocab = LabelVocabulary(Counter(labels), specials=["<pad>", "<unk>"])
    lemma2classes = dict()
    for l2s in lemma2synsetslist:
        for lemma, synsets in l2s.items():
            all_classes = lemma2classes.get(lemma, set())
            all_classes.update(
                [label_vocab.get_idx(y) for x in synsets for y in (label_mapper.get(x, [x]) if label_mapper else [x])])
            lemma2classes[lemma] = all_classes
    lemma2classes = Lemma2Synsets(data=lemma2classes)
    return get_dataset(model_name, paths, lemma2classes, mfs_file, label_mapper, label_vocab) + (label_vocab,)


def get_wnoffsets_dataset(model_name, paths, label_mapper, langs, mfs_file=None,
                          **kwargs) -> Tuple[AllennlpDataset, Lemma2Synsets, Dict, LabelVocabulary]:
    label_vocab = wnoffset_vocabulary()
    lemma2synsets = offsets_from_wn_sense_index()
    if langs is not None:
        if "en" in langs:
            langs.remove("en")
        if len(langs) > 0:
            bn2wn = get_bnoffset2wnoffset()
            bnlemma2synsets = from_bn_mapping(langs, **kwargs)
            for key, bns in bnlemma2synsets.items():
                wns = [x for y in bns for x in bn2wn[y]]
                if key in lemma2synsets:
                    lemma2synsets[key].update(wns)
                else:
                    lemma2synsets[key] = wns

    for key, synsets in lemma2synsets.items():
        lemma2synsets[key] = [label_vocab.get_idx(l) for l in synsets]

    dataset = get_dataset(model_name, paths, lemma2synsets, mfs_file, label_mapper, label_vocab)
    return dataset + (label_vocab,)


def get_bnoffsets_dataset(model_name, paths, label_mapper, langs=("en"), mfs_file=None, **kwargs) -> Tuple[
    AllennlpDataset, Lemma2Synsets, Dict]:
    lemma2synsets = from_bn_mapping(langs, **kwargs)
    label_vocab = bnoffset_vocabulary()
    for key, synsets in lemma2synsets.items():
        lemma2synsets[key] = [label_vocab.get_idx(l) for l in synsets]
    dataset = get_dataset(model_name, paths, lemma2synsets, mfs_file, label_mapper, label_vocab)
    return dataset + (label_vocab,)


def get_sensekey_dataset(model_name, paths, label_mapper, langs=None, mfs_file=None) \
        -> Tuple[AllennlpDataset, Lemma2Synsets, Dict]:
    if langs is not None:
        logger.warning(
            "[get_sensekey_dataset]: the argument langs: {} is ignored by this method.".format(",".join(langs)))

    label_vocab = wn_sensekey_vocabulary()
    lemma2synsets = sensekey_from_wn_sense_index()
    for key, synsets in lemma2synsets.items():
        lemma2synsets[key] = [label_mapper.get_idx(l) for l in synsets]
    return get_dataset(model_name, paths, lemma2synsets, mfs_file, label_mapper, label_vocab) + (label_vocab,)


class Config(dict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)

    def set(self, key, val):
        self[key] = val
        setattr(self, key, val)


def get_wnoffset2bnoffset():
    offset2bn = __load_reverse_multimap("resources/mappings/all_bn_wn.txt")
    new_offset2bn = {"wn:" + offset: bns for offset, bns in offset2bn.items()}
    return new_offset2bn


def get_bnoffset2wnoffset():
    return __load_multimap("resources/mappings/all_bn_wn.txt", value_transformer=lambda x: "wn:" + x)


def get_wnkeys2bnoffset():
    return __load_reverse_multimap("resources/mappings/all_bn_wn_keys.txt",
                                   key_transformer=lambda x: x.replace("%5", "%3"))


def get_bnoffset2wnkeys():
    return __load_multimap("resources/mappings/all_bn_wn_key.txt", value_transformer=lambda x: x.replace("%5", "%3"))


def get_wnoffset2wnkeys():
    offset2keys = dict()
    with open("/opt/WordNet-3.0/dict/index.sense") as lines:
        for line in lines:
            fields = line.strip().split(" ")
            keys = offset2keys.get(fields[1], set())
            keys.add(fields[0].replace("%5", "%3"))
            offset2keys[fields[1]] = keys
    return offset2keys


def get_wnkeys2wnoffset():
    key2offset = dict()
    with open("/opt/WordNet-3.0/dict/index.sense") as lines:
        for line in lines:
            fields = line.strip().split(" ")
            key = fields[0].replace("%5", "%3")
            pos = get_pos_from_key(key)
            key2offset[key] = ["wn:" + fields[1] + pos]
    return key2offset


def __load_reverse_multimap(path, key_transformer=lambda x: x, value_transformer=lambda x: x):
    sensekey2bnoffset = dict()
    with open(path) as lines:
        for line in lines:
            fields = line.strip().split("\t")
            bnid = fields[0]
            for key in fields[1:]:
                offsets = sensekey2bnoffset.get(key, set())
                offsets.add(value_transformer(bnid))
                sensekey2bnoffset[key_transformer(key)] = offsets
    for k, v in sensekey2bnoffset.items():
        sensekey2bnoffset[k] = list(v)
    return sensekey2bnoffset


def __load_multimap(path, key_transformer=lambda x: x, value_transformer=lambda x: x):
    bnoffset2wnkeys = dict()
    with open(path) as lines:
        for line in lines:
            fields = line.strip().split("\t")
            bnoffset = fields[0]
            bnoffset2wnkeys[key_transformer(bnoffset)] = [value_transformer(x) for x in fields[1:]]
    return bnoffset2wnkeys


def get_pos_from_key(key):
    """
    assumes key is in the wordnet key format, i.e., 's_gravenhage%1:15:00:
    :param key: wordnet key
    :return: pos tag corresponding to the key
    """
    numpos = key.split("%")[-1][0]
    if numpos == "1":
        return "n"
    elif numpos == "2":
        return "v"
    elif numpos == "3" or numpos == "5":
        return "a"
    else:
        return "r"


def get_universal_pos(simplified_pos):
    if simplified_pos == "n":
        return "NOUN"
    if simplified_pos == "v":
        return "VERB"
    if simplified_pos == "a":
        return "ADJ"
    if simplified_pos == "r":
        return "ADV"
    return ""


def get_simplified_pos(long_pos):
    long_pos = long_pos.lower()
    if long_pos.startswith("n") or long_pos.startswith("propn"):
        return "n"
    elif long_pos.startswith("adj") or long_pos.startswith("j"):
        return "a"
    elif long_pos.startswith("adv") or long_pos.startswith("r"):
        return "r"
    elif long_pos.startswith("v"):
        return "v"
    return "o"


class SemEvalOutputWriter(OutputWriter):
    def __init__(self, output_file, labeldict):
        super().__init__(output_file, labeldict)

    def write(self, outs):
        predictions = outs["predictions"]
        labels = outs["labels"]
        ids = outs["ids"]
        # predictions = torch.argmax(classes, -1)
        if type(predictions) is torch.Tensor:
            predictions = predictions.flatten().tolist()
        else:
            predictions = torch.cat(predictions).tolist()
        if type(labels) is torch.Tensor:
            labels = labels.flatten().tolist()
        else:
            labels = torch.cat(labels).tolist()
        for i, p, l in zip(ids, predictions, labels):
            self.writer.write(i + "\t" + self.labeldict[p] + "\t" + self.labeldict[l] + "\n")


def build_en_bn_lexeme2synsets_mapping(output_path):
    lemmapos2gold = dict()
    wnoffset2bnoffset = get_wnoffset2bnoffset()
    with open("/opt/WordNet-3.0/dict/index.sense") as lines:
        for line in lines:
            fields = line.strip().split(" ")
            key = fields[0]
            pos = get_pos_from_key(key)
            offset = "wn:" + fields[1] + pos
            bnoffset = wnoffset2bnoffset[offset]
            lexeme = key.split("%")[0] + "#" + pos
            golds = lemmapos2gold.get(lexeme, set())
            golds.update(bnoffset)
            lemmapos2gold[lexeme] = golds
    with open(output_path, "wt") as writer:
        for lemmapos, bnids in lemmapos2gold.items():
            writer.write(lemmapos + "\t" + "\t".join(bnids) + "\n")


if __name__ == "__main__":
    build_en_bn_lexeme2synsets_mapping("resources/lexeme_to_synsets/lexeme2synsets.reliable_sources.en.txt")
# def serialise_dataset(xml_data_path, key_gold_path, vocabulary_path, tokeniser, model_name, out_file,
#                       key2bnid_path=None, key2wnid_path=None):
#     vocabulary = Vocabulary.vocabulary_from_gold_key_file(vocabulary_path, key2bnid_path=key2bnid_path)
#
#     dataset = WSDXmlInMemoryDataset(xml_data_path, key_gold_path, vocabulary,
#                                     device="cuda",
#                                     batch_size=32,
#                                     key2bnid_path=key2bnid_path,
#                                     key2wnid_path=key2wnid_path)
#
#     with open(out_file, "wb") as writer:
#         pkl.dump(dataset, writer)
