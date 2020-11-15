from src.syntagrank.syntagrank_api import SyntagRankAPI


def test_disambiguate_text():
    lang = "it"
    syntagrank = SyntagRankAPI("syntagrank_config/config.json")
    tokens = syntagrank.disambiguate_text("questo è un semplice testo d'esempio", lang=lang)
    print(tokens)


def test_disambiguate_toknes():
    lang = "it"
    syntagrank = SyntagRankAPI("syntagrank_config/config.json")
    tokens = [{"word": "questo", "lemma": "questo", "pos": "X"},
              {"word": "è", "lemma": "essere", "pos": "v", "id": "1", "isTargetWord":True},
              {"word": "un", "lemma":"un","pos":"X"},
              {"word": "semplice", "lemma":"semplice", "pos":"a", "id":"2", "isTargetWord":True},
              {"word": "testo", "lemma":"testo", "pos":"n", "id":"3", "isTargetWord":True},
              {"word": "d'", "lemma":"di", "pos":"X"},
              {"word": "esempio", "lemma":"esempio", "pos":"n", "id":"4", "isTargetWord":True}]
    tokens = syntagrank.disambiguate_tokens(tokens, lang=lang)
    print(tokens)


if __name__ == "__main__":
    test_disambiguate_toknes()
