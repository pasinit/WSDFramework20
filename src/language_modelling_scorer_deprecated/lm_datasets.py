import gzip

from allennlp.data import DatasetReader, Instance, Token, Vocabulary
from typing import Iterable
import numpy as np

from allennlp.data.fields import TextField, IndexField, MetadataField, ArrayField
from allennlp.data.iterators import BucketIterator

from src.language_modelling_scorer_deprecated.language_modelling import AutoHuggingfaceIndexer


class LMDataset(DatasetReader):
    def __init__(self, indexer, min_tokens=5, max_tokens=70, lazy=True, one_elem_per_word=False, **kwargs):
        """
        Sentence ID TAB Wikipedia Title TAB sentence TAB start index SPACE end index SPACE BN id SPACE corresponding words in the sentence TAB
        """
        super().__init__(lazy=lazy)
        self.indexer = indexer
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.one_elem_per_word = one_elem_per_word

    def text_to_instance(self, sid, tokens, indices, words, bnids):
        fields = {"tokens": TextField(tokens, token_indexers={"tokens": self.indexer}),
                  "metadata": MetadataField(
                      {"sid": sid, "indices": [(int(x[0]), int(x[1])) for x in indices], "words": words,
                       "bnids": bnids})}
        return Instance(fields)

    def _read(self, file_path: str) -> Iterable[Instance]:
        if file_path.endswith(".gz"):
            opener = gzip.open
        else:
            opener = open
        with opener(file_path, "rt") as lines:
            for line in lines:
                fields = line.strip().split("\t")
                sid, wikititle, sentence = fields[:3]
                indices = list()
                words = list()
                bnids = list()
                aux = sentence.split(" ")
                if len(aux) < self.min_tokens or len(aux) > self.max_tokens:
                    continue
                if any(len(t) == 0 for t in aux):
                    continue
                tokens = [Token(x) for x in aux]
                for elem in fields[3:]:
                    start_idx, end_idx, bnid, word = elem.split(" ")
                    indices.append((start_idx, end_idx))
                    words.append(word)
                    bnids.append(bnid)
                    if self.one_elem_per_word:
                        yield self.text_to_instance(sid, tokens, [(start_idx, end_idx)], [word], [bnid])

                if not self.one_elem_per_word:
                    yield self.text_to_instance(sid, tokens, indices, words, bnids)


if __name__ == "__main__":
    indexer = AutoHuggingfaceIndexer("xlm-mlm-100-1280", use_starting_offsets=True)
    dataset = LMDataset(indexer, one_elem_per_word=True)
    path = "data/wikipedia_sentences/wiki.it.cleanSplitSent.pasini.txt.idx.gz"
    counter = 0
    for x in dataset.read(path):
        print(x["tokens"], x["metadata"].metadata)
        counter += 1
        if counter == 100:
            break
    iterator = BucketIterator(
        biggest_batch_first=True,
        sorting_keys=[("tokens", "num_tokens")],
        maximum_samples_per_batch=("tokens_length", 2000),
        cache_instances=True,
        # instances_per_epoch=10
    )
    iterator.index_with(Vocabulary())
    raw_train_generator = iterator(dataset.read(path),
                                   num_epochs=1,
                                   shuffle=False)
    counter = 0
    for x in raw_train_generator:
        print(x)
        counter += 1
        if counter == 100:
            break
