from allennlp.data.token_indexers import PretrainedTransformerMismatchedIndexer
from nlp_tools.data_io.datasets import ParsedToken

from src.data.dataset_utils import get_data
from src.service.wsd_as_a_service import WSDService
import yaml
import torch

from src.utils.utils import get_model

def init_service():
    with open("config/config_dev.yaml") as reader:
        config = yaml.load(reader, Loader=yaml.FullLoader)
    trained_model_path = "/home/tommaso/dev/PycharmProjects/WSDframework/data4/models/en_semcor_wngt_bn/batchnorm_wsd_classifier_xlm-roberta-large/checkpoints/best.th"
    model_config = config["model"]
    lang2inventory, _, label_vocab = get_data("bnoffsets", [], None)
    pad_id = PretrainedTransformerMismatchedIndexer(model_config["encoder_name"])._tokenizer.pad_token_id
    model = get_model(model_config, len(label_vocab),
                      pad_id,
                      label_vocab.stoi["<pad>"],
                      device="cpu")
    model.load_state_dict(torch.load(trained_model_path, map_location="cpu"))
    model.eval()

    service = WSDService(model, model_config, lang2inventory, label_vocab, device="cpu")
    return service

def test_wsd_service():
    service = init_service()
    sentence = [ParsedToken("this", None, None), ParsedToken("example", "example", "n"),
                ParsedToken("is", "be", "v"), ParsedToken("a", None, None),
                ParsedToken("great", "great", "a"),
                ParsedToken("one", "one", "n"), ParsedToken(".", None, None)]
    outputs = service.disambiguate_tokens(sentence, "en")
    for x in outputs:
        print(x)

def load_glosses(path):
    bn2gl = dict()
    with open(path) as lines:
        for line in lines:
            fields = line.strip().split("\t")
            synset = "bn:"+fields[0]
            for f in fields[1:]:
                lang = f[-2:]
                gloss = f[:-4]
                if lang == "EN":
                    bn2gl[synset] = gloss
                    break
    return bn2gl
def interactive_test():
    service = init_service()
    synset2gloss = load_glosses("/media/tommaso/My Book/factories/output/bn.synset2glosses_wn.en-it-fr-es-de.txt")
    while True:

        sentence = input("type a sentence as follows: [lang]:[your_sentence], where [lang] is "
                         "the language ISO code (2 chars) for the sentence and [your_sentence] is whatever sentence."
                         "Mark with #1 the words in your sentence that you whish to disambiguate.\n> ")
        lang=sentence[:2].strip()
        sentence = sentence[3:].strip()
        print("[DEBUG]: ", lang, " - ", sentence)
        try:
            outputs = service.disambiguate_raw_text(sentence, lang)
            for x in outputs:
                print(x, synset2gloss.get(x[-1], ''))
        except Exception as e:
            print(e)
if __name__ == "__main__":
    interactive_test()
