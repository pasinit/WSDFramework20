from typing import List, Dict, Any

from allennlp.data import Vocabulary
from allennlp.data.token_indexers import PretrainedTransformerMismatchedIndexer
from nlp_tools.allen_data.iterators import get_bucket_iterator
from nlp_tools.data_io.data_utils import MultilingualLemma2Synsets
from nlp_tools.data_io.datasets import TokenizedSentencesDataset, ParsedSentencesDataset, ParsedToken, LabelVocabulary

from src.models.neural_wsd_models import AllenWSDModel
import stanza
import torch


class WSDService(object):
    def __init__(self, model: AllenWSDModel, model_config: Dict[str, Any],
                 lang2inventory: MultilingualLemma2Synsets,
                 label_vocab: LabelVocabulary,
                 max_tokens_per_batch=1000, device="cuda", content_pos=None,
                 stopwords=None):
        self.model: AllenWSDModel = model
        self.stanza_models_cache = dict()
        self.lang2inventory = lang2inventory
        self.label_vocabulary = label_vocab
        self.stopwords = stopwords if stopwords is not None else set()
        self.encoder_name = model_config["encoder_name"]
        self.indexers_cache = dict()
        self.device = device
        self.content_pos = content_pos if content_pos is not None else {'n', 'v', 'r', 'a'}
        self.max_tokens_per_batch = max_tokens_per_batch

    def disambiguate_tokens(self, tokens: List[ParsedToken], language="en"):
        return self.disambiguate_batched_tokens([tokens], language)[0]

    def disambiguate_batched_tokens(self, tokens: List[List[ParsedToken]], language="en"):
        indexer = self.indexers_cache.get(self.encoder_name, PretrainedTransformerMismatchedIndexer(self.encoder_name))
        self.indexers_cache[self.encoder_name] = indexer

        dataset = ParsedSentencesDataset(tokens, indexer, self.label_vocabulary,
                                         self.lang2inventory.get_inventory(language))
        dataset.index_with(Vocabulary())
        iterator = get_bucket_iterator(dataset, self.max_tokens_per_batch, is_trainingset=False, device=torch.device(self.device))
        outputs = list()
        for batch in iterator:
            batch_output = list()
            net_out = self.model(**batch, compute_accuracy=False, compute_loss=False)
            batch_logits = net_out["class_logits"]
            predictions = self.model.get_predictions(batch_logits).tolist()
            batch_lexemes = batch["lexemes"]
            for tokens, lexemes in zip(batch["sentence"], batch_lexemes):
                for token, lexeme in zip(tokens, lexemes):
                    if lexeme != '':
                        p = predictions.pop(0)
                        synset = self.label_vocabulary.get_string(p)
                        batch_output.append((token.word, lexeme, synset))
                    else:
                        batch_output.append((token.word, lexeme))

            outputs.append(batch_output)
        return outputs
    def __get_stanza_pipeline(self, language):
        if language in self.stanza_models_cache:
            return self.stanza_models_cache[language]
        config = {
            'processors': 'tokenize,pos,lemma,mwt',
            'lang': language,
            "tokenize_no_ssplit": True,
            "verbose": False,
            "use_gpu": False
        }
        nlp = stanza.Pipeline(**config)
        self.stanza_models_cache[language] = nlp
        return nlp

    def disambiguate_raw_text(self, sentence, language="en"):
        pipeline = self.__get_stanza_pipeline(language)
        doc = pipeline(sentence)
        sentences = doc.sentences
        stanza_sentence = sentences[0]
        parsed_sentence = list()
        for tok in stanza_sentence.tokens:
            tok_words = tok.words
            if "-" in tok.id:
                parsed_sentence.append(ParsedToken(tok.text, tok.text, 'X'))
            else:
                w = "_".join([w.text for w in tok_words])
                l = "_".join([w.lemma for w in tok_words])
                pos = tok_words[-1].upos
                if pos == "AUX":
                    pos = "VERB"
                parsed_sentence.append(ParsedToken(w, l, pos))
        return self.disambiguate_tokens(parsed_sentence, language)
