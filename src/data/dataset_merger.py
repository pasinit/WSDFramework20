from argparse import ArgumentParser
from typing import List
from lxml import etree
import os


def merge_datasets(xml_paths: List[str], language: str, output_dir: str):
    sources = [p.split("/")[-1].replace(".data.xml", "") for p in xml_paths]

    root = etree.Element("corpus", {"lang": language, "source": "_".join(sources)})
    new_gold = dict()
    for dataset_name, path in zip(sources, xml_paths):
        gold_path = path.replace(".data.xml", ".gold.key.txt")
        with open(gold_path) as lines:
            for line in lines:
                fields = line.strip().split(" ")
                id = dataset_name + "." + fields[0]
                new_gold[id] = " ".join(fields[1:])
    root.tail = "\n"
    root.text = "\n"
    for dataset_name, path in zip(sources, xml_paths):
        dataset_root = etree.parse(path).getroot()
        for text in dataset_root.findall("./text"):
            text.attrib["id"] = dataset_name + "." + text.attrib["id"]
            for sentence in text:
                sentence.attrib["id"] = dataset_name + "." + sentence.attrib["id"]
                for elem in sentence:
                    if elem.tag == "instance":
                        instance = elem
                        instance_id = dataset_name + "." + instance.attrib["id"]
                        if instance_id not in new_gold:
                            instance.tag = "wf"
                            del instance.attrib["id"]
                        else:
                            instance.attrib["id"] = dataset_name + "." + instance.attrib["id"]
                    else:
                        if "pos" not in elem.attrib or elem.attrib["pos"] == "":
                            elem.attrib["pos"] = "X"
                        if "id" in elem.attrib:
                            del elem.attrib["id"]
            root.append(text)

    for sentence in root.findall("./text/sentence"):
        instances = sentence.findall("./instance")
        if len(instances) == 0:
            parent = sentence.getparent()
            parent.remove(sentence)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    new_dataset_name = output_dir.split("/")[-1]
    xml_path = os.path.join(output_dir, new_dataset_name + ".data.xml")
    et = etree.ElementTree(root)
    et.write(xml_path, pretty_print=True, xml_declaration=True, encoding='UTF-8')

    with open(os.path.join(output_dir, new_dataset_name + ".gold.key.txt"), "w") as gold_writer:
        gold_writer.write("\n".join([k + " " + v for k, v in new_gold.items()]))

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--xml_paths", nargs="+", required=True)
    parser.add_argument("--language", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    merge_datasets(**vars(args))