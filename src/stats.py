from collections import Counter

from torchtext.data import Dataset
from torchtext.datasets import LanguageModelingDataset, WikiText2


def multilingual_mapping_stats(map_file, lang):
    lexemes = set()
    senses = set()
    senses_by_lexeme = Counter()
    with open(map_file) as lines:
        for line in lines:
            fields = line.strip().split("\t")
            lexeme = fields[0]
            bns = fields[1:]
            lexemes.add(lexeme)
            senses.update(bns)
            senses_by_lexeme[lexeme] += len(bns)
    print("lang={}".format(lang))
    print("unique lexeme={}".format(len(lexemes)))
    print("unique senses={}".format(len(senses)))
    polysemy = sum(senses_by_lexeme.values())/float(len(senses_by_lexeme))
    print("average polisemy={}".format(polysemy))
    print("max polisemy={}".format(max(senses_by_lexeme.values())))
    print("min polisemy={}".format(min(senses_by_lexeme.values())))

from torchtext import data as ttdata
if __name__ == "__main__":
    path = "/tmp/wikitext-2/wiki.train.tokens"



    text_field = ttdata.Field(lower=True, tokenize=lambda text: text.strip().split(), init_token="<bos>",
                              eos_token="<eos>")
    texts = list()
    fields = [("text", text_field)]
    with open(path) as lines:
        for line in lines:
            if len(line.strip()) == 0:
                continue
            texts += text_field.preprocess(line)
    examples = [ttdata.Example.fromlist([texts], fields)]
    dataset = LanguageModelingDataset(path, text_field)
    print()
    # Dataset(examples, fields)

    # for lang in "it es fr de".split(" "):
    #     multilingual_mapping_stats("/media/tommaso/My Book/factories/output/framework20/lexeme2synsets.{}.txt".format(lang), lang)
    #     print("="*40)

