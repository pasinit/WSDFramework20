from argparse import ArgumentParser
import random
random.seed(42)
from random import shuffle
from typing import List, Dict
import os

from lxml import etree
from lxml.etree import Element
# from xmlformatter import Formatter
# xmlFormatter = Formatter()
from xml.dom import minidom
def create_dataset(sentences: List[Element], num_instances: int, language: str, golds:Dict[str, str], corpus_name: str):
    root = etree.Element("corpus", attrib={"name": corpus_name,"lang": language})
    text_elem = etree.Element("text", attrib={"id":"d000"})
    # text_elem.tail = "\n"
    # text_elem.text = "\n"
    root.append(text_elem)
    added_instances = 0
    added_sentences = 0
    dataset_golds = dict()
    sources = set()
    for sentence in sentences:
        sentence_instance_count = 0
        sentence.tail=""
        sentence.text=""
        instances = sentence.findall("./instance")
        source = sentence.attrib["id"].split(".")[0]
        sources.add(source)

        sentence.attrib["id"] = f"{text_elem.attrib['id']}.s{added_sentences:03d}"
        sentence.attrib["source"] = source
        for x in sentence:
            x.tail=""
        for instance in instances:
            old_id = instance.attrib["id"]
            instance_gold = golds[old_id]
            instance.attrib["id"] = sentence.attrib["id"] + f".t{sentence_instance_count:03d}"
            added_instances += 1
            sentence_instance_count += 1
            dataset_golds[instance.attrib["id"]] = instance_gold
        text_elem.append(sentence)
        added_sentences += 1
        if added_instances >= num_instances:
            break
    root.attrib["sources"] = "_".join(sources)
    return root, dataset_golds, sentences[added_sentences:] if added_sentences < len(sentences) else []

def write_dataset(root, golds, out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    et = etree.ElementTree(root)
    et_str = etree.tostring(et)
    xml_path = os.path.join(out_dir, out_dir.split("/")[-1] + ".data.xml")

    # et.write(xml_path, pretty_print=True, xml_declaration=True, encoding='UTF-8')
    # out = xmlFormatter.format_string(et_str)
    xmlstr = minidom.parseString(et_str).toprettyxml(indent="   ", newl='\n', encoding="UTF-8")
    with open(xml_path, "w") as f:
        f.write(str(xmlstr, "utf8"))
    gold_path = os.path.join(out_dir, out_dir.split("/")[-1] + ".gold.key.txt")
    with open(gold_path, "w") as writer:
        writer.write("\n".join([k + " " + v for k, v in golds.items()]))

def split(xml_path: str, test_perc: float, language: str, out_dir: str):
    root = etree.parse(xml_path).getroot()
    num_instances = len(root.findall("./text/sentence/instance"))
    test_num_instances = int(num_instances * test_perc // 100)
    dev_num_instances = num_instances - test_num_instances

    sentences = root.findall("./text/sentence")
    shuffle(sentences)
    golds = dict()
    with open(xml_path.replace(".data.xml", ".gold.key.txt")) as lines:
        for line in lines:
            fields = line.strip().split()
            golds[fields[0]] = " ".join(fields[1:])

    test_root, test_golds, remaining_sentences = create_dataset(sentences, test_num_instances, language, golds, f"test-{language}")
    dev_root, dev_golds, remaining_sentences = create_dataset(remaining_sentences, dev_num_instances, language, golds, f"dev-{language}")

    assert len(remaining_sentences) == 0
    test_out_dir = os.path.join(out_dir, f"test-{language}")
    dev_out_dir = os.path.join(out_dir, f"dev-{language}")
    write_dataset(test_root, test_golds, test_out_dir)
    write_dataset(dev_root, dev_golds, dev_out_dir)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--xml_path", required=True, type=str)
    parser.add_argument("--test_perc", type=float, required=True)
    parser.add_argument("--language", required=True, type=str)
    parser.add_argument("--out_dir", required=True, type=str)
    args = parser.parse_args()
    split(**vars(args))

