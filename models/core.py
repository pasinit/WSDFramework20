from spacy.tokens.doc import Doc


class WSD(object):
    def __init__(self):
        pass

    def __call__(self, doc: Doc):
        """
        assumes that lemmatisation and pos tagging has been alread performed and annotated in the document object.
        :param doc: the input document
        :return: the document with added semantic annotations for each noun, verb, adjective and adverb.
        """
        pass
