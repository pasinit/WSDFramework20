from argparse import ArgumentParser
import lxml.etree as etree
from lxml.etree import iterparse
from tqdm import tqdm

from src.data.dataset_utils import get_pos_from_key, get_universal_pos, get_simplified_pos, get_wnkeys2bnoffset

def filter_instances(corpus_root, lemmas):
    ids_to_keep = set()
    for sentence_xml in tqdm(corpus_root.findall("./text/sentence")):
        keep_sentence = False
        for instance in sentence_xml.findall("./instance"):
            lemma = instance.attrib["lemma"].lower() + "#" + get_simplified_pos(instance.attrib["pos"])
            if lemma in lemmas:
                keep_sentence = True
                ids_to_keep.add(instance.attrib["id"])
            else:
                del instance.attrib["id"]
                instance.tag = "wf"
        if not keep_sentence:
            parent = sentence_xml.getparent()
            parent.remove(sentence_xml)


    return ids_to_keep

def filter_dataset(path, outpath, lemmas_path, lang, dataset_name, **kwargs):
    root = etree.Element("corpus", {"lang": lang, "source": dataset_name})
    lemmas = list(line.strip().split("\t") for line in open(lemmas_path))
    lemmas = set(l+"#"+get_simplified_pos(pos) for l, pos in lemmas)
    print("filtering {} with {} lexemes".format(lang.upper(), len(lemmas)))
    with open(outpath.replace("data.xml", "gold.key.txt"), "w") as gold_writer:
        with open(path) as lines:
            corpus_root = etree.parse(lines)
        ids_to_keep = set()
        if lemmas is not None:
            ids_to_keep = filter_instances(corpus_root, lemmas)

        with open(path.replace("data.xml", "gold.key.txt")) as gold_reader:
            lines = [line for line in gold_reader if line.split(" ")[0] in ids_to_keep]
            gold_writer.write("\n".join([x.strip() for x in lines]) + "\n")

        root.extend([t for t in corpus_root.findall("./text") if len(list(t)) > 0])
    et = etree.ElementTree(root)
    et.write(outpath, pretty_print=True, xml_declaration=True, encoding='UTF-8')

def merge_datasets2(dataset_paths, lang, dataset_name, outpath, lemmas_path=None, **kwargs):
    lemmas = None
    if lemmas_path is not None:
        lemmas = set(line.strip() for line in open(lemmas_path))
    root = etree.Element("corpus", {"lang": lang, "source": dataset_name})
    with open(outpath.replace("data.xml", "gold.key.txt"), "w") as gold_writer:
        for path in dataset_paths:
            with open(path) as lines:
                corpus_root = etree.parse(lines)
            ids_to_keep = set()
            if lemmas is not None:
                ids_to_keep = filter_instances(corpus_root, lemmas)

            with open(path.replace("data.xml", "gold.key.txt")) as gold_reader:
                lines = [line for line in gold_reader if line.split(" ")[0] in ids_to_keep]
                gold_writer.write("\n".join([x.strip() for x in lines]) + "\n")

            root.extend(corpus_root.findall("./text"))
    et = etree.ElementTree(root)
    et.write(outpath, pretty_print=True, xml_declaration=True, encoding='UTF-8')


def convert_gold_wnkey2bnoffset(input_key, output_key, **kwargs):
    key2bn = dict()
    with open("resources/mappings/all_bn_wn_keys.txt") as lines:
        for line in lines:
            fields = line.strip().split("\t")
            bn = fields[0]
            for f in fields[1:]:
                bns = key2bn.get(f, set())
                bns.add(bn)
                key2bn[f] = bns
    with open(input_key) as lines, open(output_key, "w") as writer:
        for line in lines:
            fields = line.strip().split()
            all_new_gold = list()
            for f in fields[1:]:
                bns = key2bn.get(f, key2bn.get(f.replace("%3", "%5")))
                all_new_gold.extend(bns)
            writer.write(fields[0] + " " + " ".join(all_new_gold) + "\n")


def lexical_sample2semeval2013_format(input_xml, output_xml, corpus_name, **kwargs):
    with open(input_xml) as reader:
        root = etree.parse(reader).getroot()
    corpus = etree.Element("corpus", {"lang": root.attrib["lang"], "source": corpus_name})
    docid2textxml = dict()
    sentence_counter = 0
    with open(output_xml.replace("data.xml", "gold.key.txt"), "w") as writer:
        for instance in root.findall("./lexelt/instance"):
            tokenid = instance.attrib["id"]
            docid = tokenid.split(".")[0]
            src = "semcor"
            if "docsrc" in instance.attrib:
                docid = instance.attrib["docsrc"]
                src = "onesec"
            gold_answer = instance.find("answer").attrib["senseid"]
            context_xml = instance.find("context")
            lemmapos = gold_answer.split(":")[0]
            lemma = lemmapos.split("%")[0]
            pos = get_pos_from_key(gold_answer)
            head_xml = context_xml[0]
            text_xml = docid2textxml.get(docid, etree.SubElement(corpus, "text", {"id": docid, "source": src}))
            text_pre = context_xml.text.strip().split(" ")
            head_text = head_xml.text.strip()
            text_post = head_xml.tail.strip().split(" ")

            sentence_xml = etree.SubElement(text_xml, "sentence", {"id": str(sentence_counter)})
            for tok in text_pre:
                wf = etree.SubElement(sentence_xml, "wf", dict(lemma="", pos=""))
                wf.text = tok
            instance = etree.SubElement(sentence_xml, "instance", dict(lemma=lemma, pos=get_universal_pos(pos),
                                                                       id=docid + "." + "%03d" % sentence_counter + ".000"))
            writer.write(docid + "." + "%03d" % sentence_counter + ".000 " + gold_answer + "\n")
            instance.text = head_text
            for tok in text_post:
                wf = etree.SubElement(sentence_xml, "wf", dict(lemma="", pos=""))
                wf.text = tok
            sentence_counter += 1


import os




def load_gold_keys(path):
    iid2gold = dict()
    with open(path) as lines:
        for line in lines:
            fields = line.strip().split(" ")
            iid2gold[fields[0]] = fields[1:]
    return iid2gold



def fix_onesec_en():
    data_xml_path = "data/training_data/en_training_data/onesec_original_data/onesec_testset_instances/OneSeC_EN.data.xml"
    gold_keys_path = "data/training_data/en_training_data/onesec_original_data/onesec_testset_instances/OneSeC_EN.gold.key.txt"
    outpath = "data/training_data/en_training_data/onesec_original_data/onesec_testset_instances/OneSeC_EN.data.fix.xml"
    gold_keys = dict()
    with open(gold_keys_path) as lines:
        for line in lines:
            fields = line.strip().split(" ")
            gold_keys[fields[0]] = fields[1]
    with open(outpath, "w") as writer:
        writer.write("<?xml version='1.0' encoding='utf-8'?>\n")
        writer.write("<corpus>\n")
        for event, text_elem in iterparse(data_xml_path, tag="text"):
            for instance in text_elem.findall("./sentence/instance"):
                lemma = gold_keys[instance.attrib["id"]].split("%")[0]
                instance.attrib["lemma"] = lemma
                instance.attrib["pos"] = "NOUN"
            writer.write(str(etree.tostring(text_elem, pretty_print=True, encoding="utf-8"), "utf-8") + "\n")
        writer.write("</corpus>")


def build_onesec_new_testset_instances(test_set_instances_path, already_covered_path, outpath, complete_onesec_path,
                                       lang, **kwargs):
    with open(test_set_instances_path) as lines:
        new_words = set(line.strip().replace("\t", "#").replace("NOUN", "n").lower() for line in lines)
    old_onesec_root = etree.parse(already_covered_path).getroot()
    old_onesec_gold_keys = load_gold_keys(already_covered_path.replace("data.xml", "gold.key.txt"))

    covered_words = set()
    for instance in old_onesec_root.findall("./text/sentence/instance"):
        covered_words.add(instance.attrib["lemma"].lower() + "#n")

    new_words = new_words - covered_words
    gold_key_path = complete_onesec_path.replace("data.xml", "gold.key.txt")
    gold_keys = load_gold_keys(gold_key_path)

    with open(outpath, "w") as writer, open(outpath.replace("data.xml", "gold.key.txt"), "w") as key_writer:
        writer.write("<?xml version='1.0' encoding='utf-8'?>\n")
        writer.write('<corpus lang="{}" source="onesec">\n'.format(lang))
        text_element = None
        key_to_write = dict()
        for event, element in tqdm(iterparse(complete_onesec_path, tag="instance", events=["start"])):
            # if element.attrib["id"] == "d100250.s181.000":
            #     print()
            lemmapos = (element.attrib["lemma"] + "#" + element.attrib["pos"][0]).lower()
            if lemmapos not in new_words:
                continue
            iid = element.attrib["id"]
            UPOS = get_universal_pos(lemmapos.split("#")[1])
            element.attrib["pos"] = UPOS
            text_parent = element.getparent().getparent()
            if text_element is None:
                text_element = text_parent
                golds = gold_keys[iid]
                key_to_write[iid] = " ".join(golds)
            else:
                if text_element.attrib["id"] == text_parent.attrib["id"]:
                    text_element.append(element.getparent())
                    golds = gold_keys[iid]
                    key_to_write[iid] = " ".join(golds)
                else:
                    writer.write(str(etree.tostring(text_element, pretty_print=True, encoding="utf8"), "utf-8"))
                    instances = set(instance.attrib["id"] for instance in text_element.findall("./sentence/instance"))
                    assert all([instance in key_to_write for instance in instances])
                    assert all(key in instances for key in key_to_write.keys())
                    for k, v in key_to_write.items():
                        key_writer.write(k + " " + v + "\n")
                    key_to_write = dict()
                    golds = gold_keys[iid]
                    key_to_write[iid] = " ".join(golds)
                    text_element = text_parent
        # writer.write(str(etree.tostring(text_parent, pretty_print=True, encoding="utf8"), "utf-8"))
        # for instance in old_onesec_root.findall("./text/sentence/instance"):
        #     instance.attrib["pos"] = get_universal_pos(instance.attrib["pos"].lower())
        for text in old_onesec_root.findall("./text"):
            writer.write(str(etree.tostring(text, pretty_print=True, encoding="utf8"), "utf-8"))
            for instance in text.findall("./sentence/instance"):
                iid = instance.attrib["id"]
                golds = old_onesec_gold_keys[iid]
                key_writer.write(iid + " " + " ".join(golds) + "\n")
        writer.write("</corpus>")


def fix_postag_glosses():
    lang = "en"
    print(lang)
    xml_path = "data/training_data/multilingual_training_data/babel_glosses/{}/glosses_{}.parsed.filter.data.xml".format(
        lang, lang)
    gold_path = "data/training_data/multilingual_training_data/babel_glosses/{}/glosses_{}.parsed.filter.gold.key.txt".format(
        lang, lang)
    out_path = "data/training_data/multilingual_training_data/babel_glosses/{}/glosses_{}.parsed.filter.posfix.data.xml".format(
        lang, lang)
    gold_dict = dict()
    with open(gold_path) as lines:
        for line in lines:
            fields = line.strip().split("\t")
            gold_dict[fields[0]] = fields[1]
    with open(out_path, "w") as writer:
        writer.write("<?xml version='1.0' encoding='UTF-8' ?>\n")
        writer.write("<corpus lang='{}' source='glosses_{}'>\n".format(lang, lang))
        for _, elem in iterparse(xml_path, tag="text"):
            for instance in elem.findall("./sentence/instance"):
                instance.attrib["pos"] = get_universal_pos(gold_dict[instance.attrib["id"]][-1])
            writer.write(str(etree.tostring(elem, pretty_print=True, encoding="utf-8"), "utf-8"))

        writer.write("</corpus>")


def wn2bnkeys():
    sensekey2bn = dict()
    with open("/home/tommaso/dev/PycharmProjects/DocumentWSD/resources/all_bn_wn_keys.txt") as lines:
        for line in lines:
            fields = line.strip().split("\t")
            bn = fields[0]
            for f in fields[1:]:
                bns = sensekey2bn.get(f, set())
                bns.add(bn)
                sensekey2bn[f] = bns
    with open(
            "data/training_data/multilingual_training_data/onesec/en/onesec.en.new_testset_instances.gold.key.txt") as lines, \
            open(
                "data/training_data/multilingual_training_data/onesec/en/onesec.en.new_testset_instances.gold.key.bn.txt",
                "w") as writer:
        for line in lines:
            fields = line.strip().split(" ")
            bns = sensekey2bn[fields[1]]
            writer.write(fields[0] + " " + " ".join(bns) + "\n")

def convert_keys(gold_path, out_path):
    wnkey2bn = get_wnkeys2bnoffset()
    with open(gold_path) as lines, open(out_path, "w") as writer:
        for line in lines:
            fields = line.strip().split()
            tokenid, *sensekeys = fields
            bnids = [bn for sensekey in sensekeys for bn in wnkey2bn[sensekey.replace("%5", "%3")]]
            writer.write("{} {}\n".format(tokenid, " ".join(bnids)))
    



if __name__ == "__main__":
    # fix_postag_glosses()
    # exit(0)
    parser = ArgumentParser()
    subparser = parser.add_subparsers()
    merge_parser = subparser.add_parser("merge")
    merge_parser.add_argument("--dataset_paths", nargs="+", required=True)
    merge_parser.add_argument("--lang", type=str, required=True)
    merge_parser.add_argument("--dataset_name", type=str, required=True)
    merge_parser.add_argument("--outpath", type=str, required=True)
    merge_parser.add_argument("--lemmas_path", type=str)
    merge_parser.set_defaults(func=merge_datasets2)

    filter_parser = subparser.add_parser("filter")
    filter_parser.add_argument("--path", type=str, required=True)
    filter_parser.add_argument("--lang", type=str, required=True)
    filter_parser.add_argument("--dataset_name", type=str, required=True)
    filter_parser.add_argument("--outpath", type=str, required=True)
    filter_parser.add_argument("--lemmas_path", type=str, required=True)
    filter_parser.set_defaults(func=filter_dataset)

    convert_parser = subparser.add_parser("convert-ls-se13")
    convert_parser.add_argument("--input_xml", required=True, type=str)
    convert_parser.add_argument("--output_xml", required=True, type=str)
    convert_parser.add_argument("--corpus_name", required=True, type=str)
    convert_parser.set_defaults(func=lexical_sample2semeval2013_format)

    key2bnconverter = subparser.add_parser("convert_key2bn")
    key2bnconverter.add_argument("--input_key", required=True)
    key2bnconverter.add_argument("--output_key", required=True)
    key2bnconverter.set_defaults(func=convert_gold_wnkey2bnoffset)

    onesec_extractor = subparser.add_parser("extract_from_onesec")
    onesec_extractor.add_argument("--test_set_instances_path", required=True)
    onesec_extractor.add_argument("--already_covered_path", required=True)
    onesec_extractor.add_argument("--outpath", required=True)
    onesec_extractor.add_argument("--complete_onesec_path", required=True)
    onesec_extractor.add_argument("--lang", required=True)
    onesec_extractor.set_defaults(func=build_onesec_new_testset_instances)
    args = parser.parse_args()
    args.func(**vars(args))
    # build_onesec_new_testset_instances(
    #     "/home/tommaso/Documents/data/WSD_Evaluation_Framework_2.0/Evaluation_Datasets/semeval2013_{}/id2lemmapos.txt".format(lang),
    #     "/media/tommaso/4940d845-c3f3-4f0b-8985-f91a0b453b07/WSDframework/data/training_data/onesec_testset_instances/OneSeC_{}.data.xml".format(lang.upper()),
    #     "data/training_data/onesec/{}/onesec.{}.new_testset_instances.data.xml".format(lang, lang),
    #     "data/training_data/onesec/{}/onesec.{}.all.data.xml".format(lang, lang), lang)
    # exit(1)
