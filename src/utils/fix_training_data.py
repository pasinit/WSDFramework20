import os
import shutil
from lxml import etree


def load_inventory(path):
    inventory = dict()
    with open(path) as lines:
        for line in lines:
            line = line.strip()
            fields = line.split("\t")
            inventory[fields[0]] = line
    return inventory

def load_gold_answers(path):
    golds = dict()
    with open(path) as lines:
        for line in lines:
            fields = line.strip().split(" ")
            sid = fields[0]
            golds[sid] = line.strip()
    return golds

def fix_data(inventory, xml_path, out_path):
    gold = load_gold_answers(xml_path.replace(".data.xml", ".gold.key.txt"))
    root = etree.parse(xml_path).getroot()
    counter = 0
    for instance in root.findall("./text/sentence/instance"):
        if ".bg." in xml_path and instance.attrib["id"] == "d000.s04403.t00001":
            print()
        lemma, pos = instance.attrib["lemma"], instance.attrib["pos"]
        bnid_pos = gold[instance.attrib["id"]][-1]
        if pos == "PROPN":
            pos = "NOUN"
        short_pos = pos.replace("NOUN", "n").replace("VERB", "v").replace("ADV", "r").replace("ADJ", "a")
        if short_pos != bnid_pos:
            pos = bnid_pos.replace("n", "NOUN").replace("a", "ADJ").replace("r", "ADV").replace("v", "VERB")
        instance.attrib["pos"] = pos

        lemmapos = lemma.lower() + "#" + pos
        if lemmapos in inventory:
            continue
        lemma = lemma.replace("_", "")
        lemmapos = lemma+ "#" + pos
        if lemmapos not in inventory:
            lemma = lemma.replace("=", "")
            lemmapos = lemma + "#" + pos
        if lemmapos not in inventory:
            lemma = instance.text.lower()
            lemmapos = lemma + "#" + pos
        if lemmapos not in inventory:
            print(xml_path.split("/")[-1], instance.attrib["id"], lemma, pos, lemmapos, " not in inventory!")
            iid = instance.attrib["id"]
            instance.tag = "wf"
            instance.attrib["lemma"] = ""
            instance.attrib["pos"] = ""
            del instance.attrib["id"]
            del gold[iid]
            counter += 1
            continue
        instance.attrib["lemma"] = lemma
        instance.attrib["pos"] = pos
    et = etree.ElementTree(root)
    et.write(out_path, pretty_print=True)
    instances = set([i.attrib["id"] for i in root.findall("text/sentence/instance")])
    gold_ids = set(gold.keys())
    if len(gold_ids - instances) > 0:
        print(gold_ids - instances, len(gold_ids - instances), "not in instances, removing!")
        for gi in gold_ids - instances:
            del gold[gi]
        gold_ids = gold.keys()
    assert len(gold_ids - instances) == 0 and len(instances - gold_ids) == 0 or print(gold_ids - instances, instances - gold_ids)
    poss = set([i.attrib["pos"] for i in root.findall("text/sentence/instance")])
    assert all(p in {"ADJ", "ADV", "NOUN", "VERB"} for p in poss) or print(poss)
    with open(out_path.replace(".data.xml", ".gold.key.txt"), "w") as writer:
        for key, line in gold.items():
            writer.write(line + "\n")
    print(counter, "lemmapos couldn't be found")


if __name__ == "__main__":
    out_root = "/home/tommaso/Documents/data/xl-wsd/training_datasets_fixed_2/"
    xml_dir = "/home/tommaso/Documents/data/xl-wsd/training_datasets_fixed/"
    inventory_dir = "/home/tommaso/Documents/data/xl-wsd/inventories_full/"
    semcor_path_gold = "semcor.gold.key.txt"
    wngt_ex_path_gold = "wngt_examples.gold.key.txt"
    wngt_gl_path_gold = "wngt_glosses.gold.key.txt"
    semcor_folder = "semcor_{}"
    wngt_ex_folder = "wngt_examples_{}"
    wngt_gl_folder = "wngt_glosses_{}"
    for inventory_name in os.listdir(inventory_dir):
        inventory_path = os.path.join(inventory_dir, inventory_name)
        inventory = load_inventory(inventory_path)
        lang = inventory_name.split(".")[1]
        if lang in {"ko", "zh"}:
            continue
        semcor_path_xml = os.path.join(xml_dir, semcor_folder.format(
            lang), semcor_folder.format(lang)+".data.xml")
        wngt_ex_path_xml = os.path.join(xml_dir, wngt_ex_folder.format(
            lang), wngt_ex_folder.format(lang) + ".data.xml")
        wngt_gl_path_xml = os.path.join(xml_dir, wngt_gl_folder.format(
            lang), wngt_gl_folder.format(lang) + ".data.xml")

        semcor_out_dir = os.path.join(out_root, semcor_folder.format(lang))
        if not os.path.exists(semcor_out_dir):
            os.makedirs(semcor_out_dir)
        semcor_out = os.path.join(
            semcor_out_dir, semcor_folder.format(lang)+".data.xml")

        wngt_ex_out_dir = os.path.join(out_root, wngt_ex_folder.format(lang))
        if not os.path.exists(wngt_ex_out_dir):
            os.makedirs(wngt_ex_out_dir)
        wngt_ex_out = os.path.join(
            wngt_ex_out_dir, wngt_ex_folder.format(lang)+".data.xml")

        wngt_gl_out_dir = os.path.join(out_root, wngt_gl_folder.format(lang))
        if not os.path.exists(wngt_gl_out_dir):
            os.makedirs(wngt_gl_out_dir)
        wngt_gl_out = os.path.join(
            wngt_gl_out_dir, wngt_gl_folder.format(lang)+".data.xml")

        fix_data(inventory, semcor_path_xml, semcor_out)
        fix_data(inventory, wngt_ex_path_xml, wngt_ex_out)
        fix_data(inventory, wngt_gl_path_xml, wngt_gl_out)
