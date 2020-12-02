from argparse import ArgumentParser
from lxml import etree
from collections import Counter
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("xml_path")
    parser.add_argument("inventory_path")

    args = parser.parse_args()
    xml_path = args.xml_path
    inventory_path = args.inventory_path
    inventory = dict()
    with open(inventory_path) as lines:
        for line in lines:
            fields = line.strip().split("\t")
            inventory[fields[0]] = fields[1:]
    instance_ids = set()
    with open(xml_path.replace(".data.xml", ".gold.key.txt")) as lines:
        instance_ids.update([line.strip().split()[0] for line in lines])
    unique_senses = set()
    unique_lexemes = set()
    senses = list()
    instances = 0
    polysemous_words = set()
    for instance in etree.parse(xml_path).getroot().findall("./text/sentence/instance"):
        if instance.attrib["id"] not in instance_ids:
            continue
        lexeme = instance.attrib["lemma"].lower() + "#" + instance.attrib["pos"]
        l_senses = inventory[lexeme]
        unique_senses.update(l_senses)
        senses.extend(l_senses)
        unique_lexemes.add(lexeme)
        instances += 1
        if len(l_senses) > 1:
            polysemous_words.add(lexeme)

    print(f"{xml_path.split('/')[-1]} polysemous words {len(polysemous_words)}, word type polysemy "
          f"{len(unique_senses)/len(unique_lexemes):.2f}, instance polysemy {len(senses)/instances:.2f}")


