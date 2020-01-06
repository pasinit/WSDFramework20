from collections import Counter


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


if __name__ == "__main__":
    for lang in "it es fr de".split(" "):
        multilingual_mapping_stats("/media/tommaso/My Book/factories/output/framework20/lexeme2synsets.{}.txt".format(lang), lang)
        print("="*40)

