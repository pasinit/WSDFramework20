from collections import OrderedDict, Counter
from typing import Iterable, List, Dict, Callable, Any

import numpy as np
from allennlp.data import DatasetReader, Instance, TokenIndexer, Vocabulary
from allennlp.data.fields import TextField, MetadataField, ArrayField, ListField
from allennlp.data.iterators import BucketIterator
from allennlp.data.token_indexers import PretrainedBertIndexer
from allennlp.data.tokenizers import Token
from lxml import etree
from torchtext.vocab import Vocab
from transformers import BertTokenizer

from src.data.data_structures import Lemma2Synsets
from src.data.dataset_utils import get_wnkeys2wnoffset, get_wnkeys2bnoffset, get_simplified_pos, get_pos_from_key


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


class LabelVocabulary(Vocab):
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
        return LabelVocabulary(labels, specials=["<pad>", "<unk>"])

    @classmethod
    def wnoffset_vocabulary(cls):
        offsets = list()
        with open("/opt/WordNet-3.0/dict/index.sense") as lines:
            for line in lines:
                fields = line.strip().split(" ")
                key = fields[0]
                pos = get_pos_from_key(key)
                offset = fields[1] + pos
                offsets.append(offset)
        return LabelVocabulary(Counter(sorted(offsets)), specials=["<pad>"])

    @classmethod
    def bnoffset_vocabulary(cls):
        with open("resources/vocabularies/bn_vocabulary.txt") as lines:
            bnoffsets = [line.strip() for line in lines]
        return LabelVocabulary(Counter(sorted(bnoffsets)), specials=["<pad>"])

    @classmethod
    def wn_sensekey_vocabulary(cls):
        with open("/opt/WordNet-3.0/dict/index.sense") as lines:
            keys = [line.strip().split(" ")[0] for line in lines]
        return LabelVocabulary(Counter(sorted(keys)), specials=["<pad>"])


class AllenWSDDatasetReader(DatasetReader):
    def __init__(self, tokenizer: Callable[[str], List[str]] = lambda x: x.split(),
                 token_indexers: Dict[str, TokenIndexer] = None,
                 label_vocab: LabelVocabulary = None, lemma2synsets=None,
                 key2goldid: Dict[str, str] = None, max_sentence_len: int = 64,
                 sliding_window_size: int = 32):
        super().__init__(lazy=False)
        assert token_indexers is not None and label_vocab is not None and lemma2synsets is not None
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers
        self.label_vocab = label_vocab
        self.lemma2classes = lemma2synsets
        self.key2goldid = key2goldid
        self.max_sentence_len = max_sentence_len
        self.sliding_window_size = sliding_window_size

    def _read(self, file_path: str) -> Iterable[Instance]:
        gold_file = file_path.replace(".data.xml", ".gold.key.txt")
        tokid2gold = self.load_gold_file(gold_file)
        yield from self.load_xml(tokid2gold, file_path)

    def load_gold_file(self, gold_file):
        key2gold = dict()
        with open(gold_file) as lines:
            for line in lines:
                fields = line.strip().split(" ")
                key, *gold = fields
                if self.key2goldid is not None:
                    gold = [self.key2goldid[g] for g in gold]
                key2gold[key] = gold
        return key2gold

    def load_xml(self, tokid2gold, file_path):
        root = etree.parse(file_path)
        for sentence in root.findall("./text/sentence"):
            words = list()
            lemmaposs = list()
            ids = list()
            labels = list()
            for elem in sentence:
                words.append(Token(elem.text, lemma_=elem.attrib["lemma"], pos_=elem.attrib["pos"]))
                lemmaposs.append(elem.attrib["lemma"] + "#" + get_simplified_pos(elem.attrib["pos"]))
                if elem.tag == "wf":
                    ids.append(None)
                    labels.append("")
                else:
                    ids.append(elem.attrib["id"])
                    labels.append(tokid2gold[elem.attrib["id"]])

            if len(words) > self.max_sentence_len:
                for w_window, lp_window, iis_window, ls_window in self.sliding_window(words, lemmaposs, ids, labels):
                    if len(w_window) > 0 and any(x is not None for x in iis_window):
                        yield self.text_to_instance(w_window, lp_window, iis_window, np.array(ls_window))
            else:
                if any(x is not None for x in ids):
                    yield self.text_to_instance(words, lemmaposs, ids, np.array(labels))

    def sliding_window(self, words, lemmapos, ids, labels):
        for i in range(0, len(words), self.sliding_window_size):
            w_window = words[i:i + self.max_sentence_len]
            lp_window = lemmapos[i:i + self.max_sentence_len]
            is_window = ids[i:i + self.max_sentence_len]
            ls = labels[i:i + self.max_sentence_len]
            yield w_window, lp_window, is_window, ls
            if i + self.max_sentence_len > len(words):
                return

    def text_to_instance(self, input_words: List[Token], input_lemmapos: List[str], input_ids: List[str],
                         labels: np.ndarray) -> Instance:

        input_words_field = TextField(input_words, self.token_indexers)
        fields = {"tokens": input_words_field}

        id_field = MetadataField(input_ids)
        fields["ids"] = id_field

        words_field = MetadataField([t.text for t in input_words_field])
        fields["words"] = words_field

        lemmapos_field = MetadataField(input_lemmapos)
        fields["lemmapos"] = lemmapos_field

        if labels is None:
            labels = np.zeros(len(input_words_field))

        label_ids = [self.label_vocab.get_idx(l[0]) if len(l) > 0 else self.label_vocab.get_idx("<pad>") for l in labels]
        label_field = ArrayField(
            array=np.array(label_ids).astype(np.int32),
            dtype=np.long)
        fields["label_ids"] = label_field
        fields["labels"] = MetadataField([ls for ls in labels if len(ls) > 0])

        labeled_token_indices = np.array([i for i, l in enumerate(labels) if l != ""], dtype=np.int64)#np.argwhere(labels != "").flatten().astype(np.int64)
        fields["labeled_token_indices"] = MetadataField(labeled_token_indices)

        labeled_lemmapos = MetadataField(np.array(input_lemmapos)[labeled_token_indices])
        fields["labeled_lemmapos"] = labeled_lemmapos
        possible_labels = list()
        for i in range(len(input_lemmapos)):
            if input_ids[i] is None:
                continue
            classes = self.lemma2classes.get(input_lemmapos[i], [])
            classes = np.array(list(classes))

            possible_labels.append(classes)
        assert len(labeled_lemmapos) == len(labeled_token_indices) == len(possible_labels)

        possible_labels_field = MetadataField(possible_labels)
        fields["possible_labels"] = possible_labels_field

        return Instance(fields)

    @staticmethod
    def get_wnoffsets_dataset(indexers: Dict[str, Any], sliding_window=32, max_sentence_token=64):
        label_vocab = LabelVocabulary.wnoffset_vocabulary()
        lemma2synsets = Lemma2Synsets.offsets_from_wn_sense_index()
        for key, synsets in lemma2synsets.items():
            lemma2synsets[key] = [label_vocab.get_idx(l) for l in synsets]
        return AllenWSDDatasetReader(None, indexers, label_vocab=label_vocab,
                                     lemma2synsets=lemma2synsets,
                                     key2goldid=get_wnkeys2wnoffset(),
                                     max_sentence_len=max_sentence_token,sliding_window_size=sliding_window), lemma2synsets, label_vocab

    @staticmethod
    def get_bnoffsets_dataset(indexers: Dict[str, Any], sliding_window=32, max_sentence_token=64):
        lemma2synsets = Lemma2Synsets.from_bn_mapping()
        label_vocab = LabelVocabulary.bnoffset_vocabulary()
        for key, synsets in lemma2synsets.items():
            lemma2synsets[key] = [label_vocab.get_idx(l) for l in synsets]
        return AllenWSDDatasetReader(None, indexers, label_vocab=label_vocab,
                                     lemma2synsets=lemma2synsets,
                                     key2goldid=get_wnkeys2bnoffset(),
                                     max_sentence_len=max_sentence_token,sliding_window_size=sliding_window), lemma2synsets, label_vocab

    @staticmethod
    def get_sensekey_dataset(indexers: Dict[str, Any], sliding_window=32, max_sentence_token=64):
        lemma2synsets = Lemma2Synsets.sensekey_from_wn_sense_index()
        label_vocab = LabelVocabulary.wn_sensekey_vocabulary()
        for key, synsets in lemma2synsets.items():
            lemma2synsets[key] = [label_vocab.get_idx(l) for l in synsets]
        return AllenWSDDatasetReader(None, indexers, label_vocab=label_vocab,
                                     lemma2synsets=lemma2synsets,
                                     max_sentence_len=max_sentence_token,sliding_window_size=sliding_window), lemma2synsets, label_vocab


class Config(dict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)

    def set(self, key, val):
        self[key] = val
        setattr(self, key, val)


if __name__ == "__main__":
    label_vocab = LabelVocabulary.vocabulary_from_gold_key_file(
        "/home/tommaso/Documents/data/WSD_Evaluation_Framework/Training_Corpora/SemCor/semcor.gold.key.txt")
    token_indexer = PretrainedBertIndexer(
        pretrained_model="bert-base-cased",
        max_pieces=500,
        do_lowercase=True,
        use_starting_offsets=True
    )

    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

    reader = AllenWSDDatasetReader(None, {"tokens": token_indexer}, label_vocab=label_vocab)
    train_ds = reader.read(
        "/home/tommaso/Documents/data/WSD_Evaluation_Framework/Training_Corpora/SemCor/semcor.data.xml")

    iterator = BucketIterator(batch_size=1,
                              biggest_batch_first=True,
                              sorting_keys=[("tokens", "num_tokens")],
                              )
    vocab = Vocabulary()
    iterator.index_with(vocab)
    batch = next(iter(iterator(train_ds)))
    print(batch)
    print()
