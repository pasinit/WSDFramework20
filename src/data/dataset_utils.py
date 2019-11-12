import torch
from allennlp_mods.callbacks import OutputWriter


def get_wnoffset2bnoffset():
    return __load_reverse_multimap("resources/mappings/all_bn_wn.txt")


def get_bnoffset2wnoffset():
    return __load_multimap("resources/mappings/all_bn_wn.txt")


def get_wnkeys2bnoffset():
    return __load_reverse_multimap("resources/mappings/all_bn_wn_keys.txt", key_transformer=lambda x: x.replace("%5", "%3"))


def get_bnoffset2wnkeys():
    return __load_multimap("resources/mappings/all_bn_wn_key.txt", value_transformer=lambda x : x.replace("%5", "%3"))


def get_wnoffset2wnkeys():
    offset2keys = dict()
    with open("/opt/WordNet-3.0/dict/index.sense") as lines:
        for line in lines:
            fields = line.strip().split(" ")
            keys = offset2keys.get(fields[1], set())
            keys.add(fields[0].replace("%5", "%3"))
            offset2keys[fields[1]] = keys
    return offset2keys


def get_wnkeys2wnoffset():
    key2offset = dict()
    with open("/opt/WordNet-3.0/dict/index.sense") as lines:
        for line in lines:
            fields = line.strip().split(" ")
            key = fields[0].replace("%5", "%3")
            pos = get_pos_from_key(key)
            key2offset[key] = fields[1] +  pos
    return key2offset


def __load_reverse_multimap(path, key_transformer=lambda x: x, value_transformer=lambda x:x):
    sensekey2bnoffset = dict()
    with open(path) as lines:
        for line in lines:
            fields = line.strip().split("\t")
            bnid = fields[0]
            for key in fields[1:]:
                offsets = sensekey2bnoffset.get(key, set())
                offsets.add(value_transformer(bnid))
                sensekey2bnoffset[key_transformer(key)] = offsets
    for k, v in sensekey2bnoffset.items():
        sensekey2bnoffset[k] = list(v)
    return sensekey2bnoffset


def __load_multimap(path, key_transformer=lambda x: x, value_transformer=lambda x:x):
    bnoffset2wnkeys = dict()
    with open(path) as lines:
        for line in lines:
            fields = line.strip().split("\t")
            bnoffset = fields[0]
            bnoffset2wnkeys[key_transformer(bnoffset)] = [value_transformer(x) for x in fields[1:]]
    return bnoffset2wnkeys


def get_pos_from_key(key):
    """
    assumes key is in the wordnet key format, i.e., 's_gravenhage%1:15:00:
    :param key: wordnet key
    :return: pos tag corresponding to the key
    """
    numpos = key.split("%")[-1][0]
    if numpos == "1":
        return "n"
    elif numpos == "2":
        return "v"
    elif numpos == "3" or numpos == "5":
        return "a"
    else:
        return "r"


def get_universal_pos(simplified_pos):
    if simplified_pos == "n":
        return "NOUN"
    if simplified_pos == "v":
        return "VERB"
    if simplified_pos == "a":
        return "ADJ"
    if simplified_pos == "r":
        return "ADV"
    return ""

def get_simplified_pos(long_pos):
    long_pos = long_pos.lower()
    if long_pos.startswith("n") or long_pos.startswith("propn"):
        return "n"
    elif long_pos.startswith("adj") or long_pos.startswith("j"):
        return "a"
    elif long_pos.startswith("adv") or long_pos.startswith("r"):
        return "r"
    elif long_pos.startswith("v"):
        return "v"
    return "o"


class SemEvalOutputWriter(OutputWriter):
    def __init__(self, output_file, labeldict):
        super().__init__(output_file, labeldict)

    def write(self, outs):
        predictions = outs["predictions"]
        labels = outs["labels"]
        ids = outs["ids"]
        # predictions = torch.argmax(classes, -1)
        if type(predictions) is torch.Tensor:
            predictions = predictions.flatten().tolist()
        else:
            predictions = torch.cat(predictions).tolist()
        if type(labels) is torch.Tensor:
            labels = labels.flatten().tolist()
        else:
            labels = torch.cat(labels).tolist()
        for i, p, l in zip(ids, predictions, labels):
            self.writer.write(i + "\t" + self.labeldict[p] + "\t" + self.labeldict[l] + "\n")


# def serialise_dataset(xml_data_path, key_gold_path, vocabulary_path, tokeniser, model_name, out_file,
#                       key2bnid_path=None, key2wnid_path=None):
#     vocabulary = Vocabulary.vocabulary_from_gold_key_file(vocabulary_path, key2bnid_path=key2bnid_path)
#
#     dataset = WSDXmlInMemoryDataset(xml_data_path, key_gold_path, vocabulary,
#                                     device="cuda",
#                                     batch_size=32,
#                                     key2bnid_path=key2bnid_path,
#                                     key2wnid_path=key2wnid_path)
#
#     with open(out_file, "wb") as writer:
#         pkl.dump(dataset, writer)
