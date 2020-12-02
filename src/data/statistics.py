import os
from collections import OrderedDict

from lxml import etree

import pandas
from tabulate import tabulate
from tqdm import tqdm

from src.data.dataset_utils import get_simplified_pos, get_pos_from_key


def load_inventory(path):
    inventory = dict()
    with open(path) as lines:
        for line in lines:
            fields = line.strip().split("\t")
            lemma, pos = fields[0].split("#")
            lexeme = lemma + "#" + get_simplified_pos(pos)
            inventory[lexeme] = fields[1:]
    return inventory


def word_types(xml_path):
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
        lexeme = instance.attrib["lemma"].lower() + "#" + get_simplified_pos(instance.attrib["pos"])
        word_types.add(lexeme)

    return {"word_types": len(word_types)}


def polysemy(xml_path, inventory_path, pos=None):
    inventory = load_inventory(inventory_path)
    instance_ids = set()
    with open(xml_path.replace(".data.xml", ".gold.key.txt")) as lines:
        instance_ids.update([line.strip().split()[0] for line in filter(lambda x: get_pos_from_key(x.strip()) == pos
        if not "bn:" in x else x.strip()[-1] == pos, lines)])
    unique_lexemes = set()
    unique_senses = set()
    senses = list()
    instances = 0
    missing = set()
    polysemous_words = set()
    for instance in etree.parse(xml_path).getroot().findall("./text/sentence/instance"):
        if instance.attrib["id"] not in instance_ids:
            continue
        lexeme = instance.attrib["lemma"].lower() + "#" + get_simplified_pos(instance.attrib["pos"])
        if lexeme not in inventory:
            missing.add(lexeme)
            continue
        l_senses = inventory[lexeme]  # , set())
        unique_senses.update(l_senses)
        senses.extend(l_senses)
        unique_lexemes.add(lexeme)
        instances += 1
        if len(l_senses) > 1:
            polysemous_words.add(lexeme)
    # print("missing", len(missing))
    return {"polysemous words": len(polysemous_words),
            "word type polysemy": len(unique_senses) / max(len(unique_lexemes), 1),
            "instance polysemy": len(senses) / max(instances, 1)}


def languages():
    return "bg, ca, da, de, en, es, en-coarse, en-no-sem10-no-sem07, et, eu, fr, gl, hr, hu, it, ja, ko, nl, sl, zh".split(
        ", ")


def dataset_stats(data_xml_path, inventory_path, key2bn=None):
    stats = dict()
    stats.update(word_types(data_xml_path))
    stats.update(polysemy(data_xml_path, inventory_path))
    stats.update(num_instances_and_unique_synsets(data_xml_path, key2bn))
    return stats


def num_instances_and_unique_synsets(data_xml_path, key2bn=None):
    gold_path = data_xml_path.replace(".data.xml", ".gold.key.txt")
    with open(gold_path) as lines:
        instances = [l.strip() for l in lines.readlines()]
    synsets = set([j if key2bn is None else key2bn[j] for i in instances for j in i.split(" ")[1:]])
    return {"instances": len(instances), "unique_synsets": len(synsets)}


def load_key2bn():
    key2bn = dict()
    with open("/home/tommaso/dev/PycharmProjects/WSDframework/resources/mappings/all_bn_wn_keys.txt") as lines:
        for line in lines:
            fields = line.strip().split("\t")
            for key in fields[1:]:
                bns = key2bn.get(key, set())
                bns.add(fields[0])
                key2bn[key] = fields[0]
    return key2bn


def compute_stats(test_dir, inventory_dir):
    all_stats = dict()
    for dataset_dir in tqdm(os.listdir(test_dir)):
        if dataset_dir == "complete_datasets":
            continue
        lang = dataset_dir.split("-")[-1]
        if lang != "et":
            continue
        if lang == "sem07" or lang == "coarse":
            lang = "en"
        key2bn = None
        if lang == "en":
            key2bn = load_key2bn()
        inventory_path = os.path.join(inventory_dir, f"inventory.{lang}.withgold.txt")
        data_path = os.path.join(test_dir, dataset_dir, dataset_dir + ".data.xml")
        stats = dataset_stats(data_path, inventory_path, key2bn)
        all_stats[dataset_dir] = stats
    d = OrderedDict()
    formatters = {"unique_synsets": lambda x: str(int(x)),
                  "instances": lambda x: str(int(x)),
                  "instance polysemy": lambda x: f"{x:.3f}",
                  "word type polysemy": lambda x: f"{x:.3f}",
                  "polysemous words": lambda x: f"{x:d}",
                  "word_types": lambda x: f"{x:d}"
                  }
    for lang in sorted(languages()):
        test_stats, dev_stats, train_stats = None, None, None
        if f"test-{lang}" in all_stats:
            test_stats = all_stats[f"test-{lang}"]
        if f"dev-{lang}" in all_stats:
            dev_stats = all_stats[f"dev-{lang}"]
        if f"train-{lang}" in all_stats:
            train_stats = all_stats[f"train-{lang}"]
        if test_stats is None and dev_stats is None and train_stats is None:
            continue

        stats = dict()
        keys = test_stats.keys() if test_stats is not None else dev_stats.keys() if dev_stats is not None else train_stats.keys()
        for k in keys:
            formatter = formatters[k]
            tv, dv, trv = None, None, None
            if test_stats is not None:
                tv = formatter(test_stats[k])
            if dev_stats is not None:
                dv = dev_stats.get(k, None) if dev_stats is not None else None
                dv = formatter(dv)
            if train_stats is not None:
                trv = formatter(train_stats[k])
            out = list()
            if trv is not None:
                out.append(trv)
            if tv is not None:
                out.append(tv)
            if dv is not None:
                out.append(dv)

            stats[k] = out

        d[lang] = stats
    df = pandas.DataFrame.from_dict(d).T
    with pandas.option_context('display.max_rows', None, 'display.max_columns',
                               None):  # , 'display.float_format', '{:0.3f}'.format):
        print(df.to_csv(sep="\t"))
        print(tabulate(df))
        print(df.to_latex())
    header = d.keys()
    # print("\t".join(header))

    # for k, d in d.items():
    #     line = "\t".join([f"{x}\t{y}" for x, y in d.values()])
    #     print(k + "\t" + line)


if __name__ == "__main__":
    # compute_stats(
    #     "/home/tommaso/dev/PycharmProjects/WSDframework/data2/training_data/multilingual_training_data/TranslatedTrainESCAPED_merged",
    #     "/home/tommaso/Documents/data/WSD_Evaluation_Framework_3.0/inventories")
    # compute_stats("/home/tommaso/dev/PycharmProjects/WSDframework/data2/training_data/en_training_data/semcor+michele_wngt",
    #               "/home/tommaso/Documents/data/WSD_Evaluation_Framework_3.0/inventories")
    root = "/home/tommaso/Documents/data/WSD_Evaluation_Framework_3.0/new_evaluation_datasets/"
    print("NOUN\tVERB\tADJ\tADV")
    for test_name in "en en-no-sem10-no-sem07 en-coarse eu bg ca zh hr da nl et fr gl de hu it ja ko sl es".split():
        folder_name = "test-" + test_name
        lang = test_name.split("-")[0]
        data_file = os.path.join(root, folder_name, folder_name + ".data.xml")
        polysemies = list()
        for pos in "n v a r".split():
            stats = polysemy(data_file,
                             "/home/tommaso/Documents/data/WSD_Evaluation_Framework_3.0/inventories/inventory.{}.withgold.txt".format(
                                 lang),
                             pos=pos)
            instance_polysemy = stats["word type polysemy"]
            polysemies.append(instance_polysemy)
        print(folder_name + "\t" + "\t".join([str(x) for x in polysemies]))
