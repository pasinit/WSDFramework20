from lxml import etree
import os

from lxml.etree import ElementTree
from tqdm import tqdm

from src.data.data_preparation import load_gold_keys
from src.data.dataset_utils import wn_sensekey_vocabulary, sensekey_from_wn_sense_index, get_simplified_pos


def build(datasets, out_dir, inventory):
    for dataset in tqdm(datasets):
        new_instances = 0
        instance2gold = load_gold_keys(dataset.replace(".data.xml", ".gold.key.txt"))
        out_data_path = os.path.join(out_dir, dataset.split("/")[-1])
        out_gold_path = out_data_path.replace(".data.xml", ".gold.key.txt")
        root = etree.parse(dataset).getroot()
        for sentence in root.findall("./text/sentence"):
            sid = sentence.attrib["id"]
            for token in sentence:

                if token.tag == "wf":
                    lemma = token.attrib["lemma"]
                    pos = token.attrib["pos"]
                    pos = get_simplified_pos(pos)
                    senses = inventory.get(lemma + "#" + pos, None)

                    if senses is not None and len(senses) == 1:
                        mono_id = "monosemous." + sid + "." + str(new_instances)
                        new_instances += 1
                        instance2gold[mono_id] = list(senses)[0]
                        token.tag = "instance"
                        token.attrib["id"] = mono_id
        ElementTree(root).write(out_data_path, pretty_print=True, xml_declaration=True, encoding="utf-8")
        with open(out_gold_path, "w") as writer:
            for instance, synsets in instance2gold.items():
                writer.write(instance + " " + " ".join(synsets) + "\n")


if __name__ == "__main__":
    inventory = sensekey_from_wn_sense_index()
    datasets = ["/home/tommaso/dev/PycharmProjects/WSDframework/data2/training_data/en_training_data/wngt_michele/wngt_michele_examples/wngt_michele_examples.data.xml",
                "/home/tommaso/dev/PycharmProjects/WSDframework/data2/training_data/en_training_data/wngt_michele/wngt_michele_glosses/wngt_michele_glosses.data.xml",
                "/media/tommaso/4940d845-c3f3-4f0b-8985-f91a0b453b07/WSDframework/data/training_data/en_training_data/semcor/semcor.data.xml"]
    out_dir = "/media/tommaso/4940d845-c3f3-4f0b-8985-f91a0b453b07/WSDframework/data/training_data/en_training_data/michele_trainingset/"
    build(datasets, out_dir, inventory)