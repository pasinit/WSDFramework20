from argparse import ArgumentParser
from lxml import etree

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("xml_path")
    args = parser.parse_args()
    xml_path = args.xml_path
    gold_path = xml_path.replace(".data.xml", ".gold.key.txt")
    instance_ids = set()
    with open(gold_path) as lines:
        for line in lines:
            id, *_ = line.strip().split()
            instance_ids.add(id)
    word_types = set()
    for instance in etree.parse(xml_path).getroot().findall("./text/sentence/instance"):
        if instance.attrib["id"] not in instance_ids:
            continue
        lexeme = instance.attrib["lemma"].lower() + "#" + instance.attrib["pos"]
        word_types.add(lexeme)
    print(xml_path.split("/")[-1] + ":" + str(len(word_types)))