import json
from typing import List, Dict
import requests

from src.misc.wsdlogging import get_info_logger

logger = get_info_logger(__name__)


class AnnotatedToken(object):
    def __init__(self, token_id: int, text: str, lemma: str, pos: str, sense_id: str, position: Dict[str, int]):
        self.text = text
        self.lemma = lemma
        self.pos = pos
        self.token_id = token_id
        self.sense_id = sense_id
        self.start_offset = position["charOffsetBegin"] if position else -1
        self.end_offset = position["charOffsetEnd"] if position else -1

    @classmethod
    def from_json(cls, data):
        lemma = None if "lemma" not in data else data["lemma"]
        pos = None if "pos" not in data else data["pos"]
        token_id = None if "token_id" not in data else data["token_id"]

        return AnnotatedToken(token_id, data["word"], lemma, pos, data.get("senseID", None), data.get("position", None))

    def to_json(self):
        aux = {"text": self.text, "senseID": self.sense_id,
               "charOffsetBegin": self.start_offset,
               "charOffsetEnd": self.end_offset,
               "lemma": self.lemma, "token_id": self.token_id, "pos": self.pos}

        return json.dumps(aux, ensure_ascii=False)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return self.to_json()


class AnnotatedText(tuple):

    @classmethod
    def from_json(cls, data):
        aux = list()
        for d in data:
            aux.append(AnnotatedToken.from_json(d))
        return AnnotatedText(aux)


class SyntagRankAPI(object):
    class __SyntagRankAPI(object):
        def __init__(self, config, wn2bn=None):
            with open(config) as reader:
                self.config = json.load(reader)
            self.wn2bn = wn2bn

        def __maybe_error(self, r, answer):
            if r.status_code != 200:
                logger.error(f"status: {answer['status']} message: {answer['message']}")
                raise RuntimeError(f"status: {answer['status']} message: {answer['message']}")

        def disambiguate_text(self, text: str, lang="EN"):
            lang = lang.upper()
            url = self.config["url"] + self.config["disambiguate_text_endpoint"]
            payload = {"lang": lang, "text": text}
            r = requests.get(url, params=payload)
            answer = json.loads(r.text)
            self.__maybe_error(r, answer)
            for annotated_token in answer["tokens"]:
                positions = annotated_token["position"]
                start, end = positions["charOffsetBegin"], positions["charOffsetEnd"]
                text_span = text[start:end]
                annotated_token["word"] = text_span
                if self.wn2bn is not None:
                    annotated_token["senseID"] = self.wn2bn[annotated_token["senseID"]]
            tokens = answer["tokens"]
            return AnnotatedText.from_json(tokens)

        def disambiguate_tokens(self, tokens: List[Dict[str, str]], lang="EN"):
            """
            :param
                tokens: list of tokens. Each tokens has the information on the word, the lemma, the pos, the token_id
                and whether it is a target_word or not.
            :return
                a list of AnnotatedToken.
            """
            lang = lang.upper()
            url = self.config["url"] + self.config["disambiguate_tokens_endpoint"]
            id2token_idx = dict()
            for idx, token in enumerate(tokens):
                assert "lemma" in token
                assert "pos" in token
                assert "word" in token
                if "isTargetWord" not in token and "is_target_word" not in token:
                    token["isTargetWord"] = False
                if "id" not in token:
                    token["id"] = "None"
                if "is_target_word" in token:
                    token["isTargetWord"] = token["is_target_word"]
                    del token["is_target_word"]
                id2token_idx[token["id"]] = idx
            payload = {"lang": lang, "words": tokens}
            r = requests.post(url, json=payload)
            response = json.loads(r.text)
            self.__maybe_error(r, response)
            for tagged_token in response["result"]:
                id = tagged_token["id"]
                idx = id2token_idx[id]
                synset = tagged_token["synset"]
                if self.wn2bn is not None:
                    synset = self.wn2bn[synset]
                tokens[idx]["senseID"] = synset
                tokens[idx]["token_id"] = id
            return AnnotatedText.from_json(tokens)

    instance = None

    def __init__(self, config, wn2bn:Dict[str, str]=None):
        if not SyntagRankAPI.instance:
            SyntagRankAPI.instance = SyntagRankAPI.__SyntagRankAPI(config, wn2bn)

    def __getattr__(self, name):
        return getattr(self.instance, name)

    def disambiguate_text(self, text: str, lang="EN"):
        return self.instance.disambiguate_text(text, lang)

    def disambiguate_tokens(self, tokens, lang):
        return self.instance.disambiguate_tokens(tokens, lang)
