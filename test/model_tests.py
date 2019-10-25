from data.data_structures import Lemma2Synsets
from data.datasets import Vocabulary
from models.neural_wsd_models import TransformerFFWSDModel

if __name__ == "__main__":
    lemma2synsetpath = "/media/tommaso/My Book/factories/output/lemma2bnsynsets.wn_part.en.txt"
    key_gold_path = "/home/tommaso/Documents/data/WSD_Evaluation_Framework/Training_Corpora/SemCor/semcor.gold.key.txt"
    vocabulary = Vocabulary.vocabulary_from_gold_key_file(key_gold_path, key2bnid_path="/home/tommaso/dev/PycharmProjects/DocumentWSD/resources/all_bn_wn_keys.txt")
    lemma2synsets = Lemma2Synsets(lemma2synsetpath)
    wsd_model = TransformerFFWSDModel("bert-base-cased", "cuda", 768, len(vocabulary), vocabulary, lemma2synsets=lemma2synsets)
    input_x = [["Bianca", "is", "useless"], ["All", "work", "and", "no", "play", "makes", "Tommaso", "a", "dull", "boy"]]
    input_lemmas = [["Bianca#n", "be#v", "useless#a"], ["All#a", "work#n", "and#o", "no#r", "play#n", "make#v", "Tommaso#n", "a#o", "dull#a", "boy#n"]]
    output = wsd_model(input_x, input_lemmas)
    for sentence_i, output_i in zip(input_lemmas, output):
        for token, out in zip(sentence_i, output_i):
            print(token, out)
        print("="*10)