from typing import List, Dict

from src.misc.wsdlogging import get_info_logger
from src.syntagrank.syntagrank_api import SyntagRankAPI, AnnotatedText, AnnotatedToken
from lxml import etree
from tqdm import tqdm

logger = get_info_logger(__name__)


def write_predictions(disambiguations: List[AnnotatedToken], output_path: str):
    with open(output_path, "w") as writer:
        for disambiguation in disambiguations:
            writer.write("{} {}\n".format(disambiguation.token_id, disambiguation.sense_id))


def tag_with_syntagrank(syntagrank: SyntagRankAPI, xml_path: str, lang: str, output_path: str):
    root = etree.parse(xml_path).getroot()
    disambiguations = list()
    for sentence in tqdm(root.findall("./text/sentence")):
        has_instance = False
        tokens = []
        for token in sentence:
            attribs = token.attrib
            id = attribs.get("id", None)
            token_dict = {"word": token.text, "lemma": attribs.get("lemma", ''),
                          "pos": attribs.get("pos", '')}
            if token.tag == "instance":
                token_dict["id"] = id
                token_dict["isTargetWord"] = True

            if "id" in attribs:
                has_instance = True
            tokens.append(token_dict)
        if has_instance:
            disambiguated_tokens = syntagrank.disambiguate_tokens(tokens, lang)
            disambiguations.extend(
                [x for x in disambiguated_tokens if x.token_id is not None and x.sense_id is not None])
    logger.info(xml_path.split("/")[-1] + f" ({len(disambiguations)} instances) disambiguated")
    write_predictions(disambiguations, output_path)


def load_wn2bn(path):
    wn2bn = dict()
    with open(path) as lines:
        for line in lines:
            fields = line.strip().split("\t")
            for wn in fields[1:]:
                wn2bn["wn:" + wn] = fields[0]
    return wn2bn


if __name__ == "__main__":
    wn2bn_path = "/home/tommaso/dev/PycharmProjects/WSDframework/resources/mappings/all_bn_wn.txt"
    wn2bn = load_wn2bn(wn2bn_path)
    syntagrank = SyntagRankAPI("syntagrank_config/config.json", wn2bn)
    datasets = "senseval2 senseval3 semeval2007 semeval2007-coarse semeval2010 semeval2013 semeval2015 ALL-no-semeval2007 semeval2010-it semeval2013-it semeval2015-it wordnet-italian semeval2013-es semeval2015-es wordnet-spanish semeval2013-fr semeval2013-de".split()
    langs = "en en en en en en en en it it it it es es es".split()
    for dataset,lang in zip(datasets[5:8], langs[5:8]):
        # if "-" in dataset:
        #     lang = dataset.split("-")[-1]
        #     lang = lang[:2]
        # else: lang = "en"
        #
        # if lang == "sp":
        #     lang = "es"
        logger.info(dataset)
        tag_with_syntagrank(syntagrank,
                            f"/home/tommaso/Documents/data/WSD_Evaluation_Framework_3.0/Evaluation_Datasets/{dataset}/{dataset}.data.xml",
                            lang,
                            f"data4/models/syntagrank/evaluation/{dataset}.predictions.txt")
