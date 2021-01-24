import os
import pickle as pkl
import random

import torch
from lxml import etree
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizer, LxmertTokenizer

from src.data.dataset_utils import get_pos_from_key, get_simplified_pos
import logging

LEVEL = logging.DEBUG
logger = logging.getLogger(__name__)
ch = logging.StreamHandler()
ch.setLevel(LEVEL)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)
logger.setLevel(LEVEL)

random.seed(34)
import numpy as np

CACHE_DIR = ".cache/"
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)
fake_features = np.zeros([36, 2048])
fake_pos = np.zeros([36, 4])


# def read_imgid_index(path):
#     index = dict()
#     with open(path) as lines:
#         for line in lines:
#             sentenceid, imgid = line.strip().split("\t")
#             if len(sentenceid.split(".")) > 2:
#                 sentenceid = ".".join(sentenceid.split(".")[:2])
#             index[sentenceid] = int(imgid)
#     return index

def read_imgid_index(path):
    index = dict()
    with open(path) as lines:
        for line in lines:
            sentence_id, *img_ids_scores = line.strip().split("\t")
            img_id = img_ids_scores[0].split(":")[0]
            index[sentence_id] = img_id
    return index


def get_imgid_2_arrindex(img_ids):
    index = dict()
    for i, img_id in enumerate(img_ids):
        index[img_id] = i
    return index


def read_wn_inventory(path):
    inventory = dict()
    with open(path) as lines:
        for line in lines:
            fields = line.strip().split(" ")
            sensekey = fields[0]
            pos = get_pos_from_key(sensekey)
            lemmapos = sensekey.split("%")[0] + "#" + pos
            choices = inventory.get(lemmapos, set())
            choices.add(sensekey)
            inventory[lemmapos] = choices
    return inventory


def read_gold_keys(path):
    golds = dict()
    with open(path) as lines:
        for line in lines:
            fields = line.strip().split(" ")
            golds[fields[0]] = set(fields[1:])
    return golds


import hashlib


class MultimodalTxtDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer,
                 txt_path: str,
                 sentenceid_2_imgid_path: str,
                 img_features_files,
                 sense_index_path: str,
                 use_cache: bool = True):
        self.path_xml = txt_path
        self.path_gold = txt_path.replace(".data.xml", ".gold.key.txt")
        self.name = txt_path.split("/")[-1]
        self.examples = list()
        self.pad_token_id = tokenizer.pad_token_id
        self.sense2class = dict()
        self.inventory = read_wn_inventory(sense_index_path)
        with open(sense_index_path) as lines:
            for i, line in enumerate(lines):
                fields = line.strip().split(" ")
                self.sense2class[fields[0]] = i
        self.class2sense = dict([list(reversed(x)) for x in self.sense2class.items()])
        gold_path = txt_path.replace(".data.xml", ".gold.key.txt")
        token_id_2_gold = read_gold_keys(gold_path)
        dataset_id = str(tokenizer.__class__) + self.path_xml + img_features_files.fid.name + sentenceid_2_imgid_path
        self.dataset_id = hashlib.md5(bytes(dataset_id, "utf8")).hexdigest()
        self.dataset_cached_path = os.path.join(CACHE_DIR, self.dataset_id)
        if os.path.exists(self.dataset_cached_path) and use_cache:
            print("found CACHE", self.dataset_cached_path)
            with open(self.dataset_cached_path, "rb") as reader:
                self.examples = pkl.load(reader)
                logger.info("dataset loaded from cache {}".format(self.dataset_cached_path))
        else:
            examples = self.load_xml(txt_path, token_id_2_gold, tokenizer)
            self.add_img_features(examples, img_features_files, sentenceid_2_imgid_path)
            self.examples = examples
            if use_cache:
                with open(self.dataset_cached_path, "wb") as writer:
                    pkl.dump(examples, writer)
                    logger.info("dataset dumped in cache {}".format(self.dataset_cached_path))

    def add_fakeimg_features(self, examples, img_features_path, sentenceid_2_imgid_path):
        for example in examples:
            example["img_boxes"] = fake_pos
            example["img_features"] = fake_features

    def add_img_features(self, examples, img_features_files, sentenceid_2_imgid_path):
        if img_features_files is None:
            self.add_fakeimg_features(examples, None, None)
            return
        logger.info("loading images' features")
        all_img_features, all_img_boxes = img_features_files["features"], img_features_files["normalized_boxes"]
        logger.info("images' features loaded")

        img_ids = img_features_files["image_ids"]
        sentenceid2imgid = read_imgid_index(sentenceid_2_imgid_path)
        img2arrindex = get_imgid_2_arrindex(img_ids)
        ids_not_found = set()
        for example in examples:
            example_id = example["sentence_id"]
            imgid = sentenceid2imgid.get(example_id, None)
            if imgid is None:
                ids_not_found.add(example_id)
            img_idx = img2arrindex.get(imgid, None)
            if img_idx is None:
                example["img_boxes"] = fake_pos,
                example["img_features"] = fake_features
                continue

            img_features, img_boxes = all_img_features[img_idx], all_img_boxes[img_idx]
            example["img_boxes"] = img_boxes
            example["img_features"] = img_features
        print(len(ids_not_found), "ids not found.")
    def load_xml(self, path, id2golds, tokenizer):
        root = etree.parse(path).getroot()
        corpus_name = root.attrib["source"]
        examples = list()
        for sentence in tqdm(root.findall("./text/sentence")):
            token2segment_ids = dict()
            aux = tokenizer.encode("t", add_special_tokens=True)
            start_token, end_token = aux[0], aux[-1]
            segment_ids = [start_token]
            tokens = list()
            lexemes = []
            labels_mask = [0]
            labels = []
            choices = []
            indexed_choices = []
            instance_ids = []
            for i, token in enumerate(sentence):
                word = token.text
                token2segment_ids[i] = (len(segment_ids))
                word_ids = tokenizer.encode(word, add_special_tokens=False)
                token2segment_ids[i] = (len(segment_ids), len(segment_ids) + len(word_ids))
                segment_ids.extend(word_ids)
                tokens.append(word)
                if token.tag == "instance":
                    instance_ids.append(corpus_name + "_" + token.attrib["id"])
                    pos = get_simplified_pos(token.attrib["pos"])
                    lexeme = token.attrib["lemma"] + "#" + pos
                    lexemes.append(lexeme)
                    instance_id = token.attrib["id"]
                    golds = id2golds[instance_id]
                    labels.append({self.sense2class[x] for x in golds})
                    l_choices = self.inventory.get(lexeme, None)
                    choices.append(l_choices)
                    if l_choices is not None:
                        indexed_choices.append({self.sense2class[x] for x in l_choices})
                    else:
                        indexed_choices.append(None)
                    labels_mask.append(1)
                    for _ in range(len(word_ids) - 1):
                        labels_mask.append(0)
                else:
                    for _ in range(len(word_ids)):
                        labels_mask.append(0)
            labels_mask.append(0)
            segment_ids.append(end_token)
            if len(labels) == 0:
                continue
            examples.append(
                {"sentence_id": sentence.attrib["id"],
                 "tokens": tokens,
                 "instance_ids": instance_ids,
                 "input_ids": segment_ids,
                 "token2segment_ids": token2segment_ids,
                 "lexemes": lexemes,
                 "labels_mask": labels_mask,
                 "labels": labels,
                 "choices": choices,
                 "indexed_choices": indexed_choices})
            # if len(examples) == 120:
            #     break
        return examples

    def __getitem__(self, idx):
        return self.examples[idx]

    def __len__(self):
        return len(self.examples)

    def get_image_features(self, synset_id, img_features_index, img_features, img_boxes):
        if synset_id not in img_features_index:
            return None, None
        indices = list(img_features_index[synset_id])
        idx = random.randint(0, len(indices) - 1)
        img_index = indices[idx]
        features = img_features[img_index]
        boxes = img_boxes[img_index]
        return boxes, features

    def get_batch_fun(self):
        def collate_fn(examples):
            input_ids, img_features, img_pos = zip(
                *[(
                    torch.Tensor(e["input_ids"]).long(), torch.Tensor(e["img_features"]).squeeze(),
                    torch.Tensor(e["img_boxes"]).squeeze())
                    for e in examples])
            labels = []
            labels_mask = []
            indexed_choices = []
            choices = []
            instance_ids = []
            for e in examples:
                e_labels = e["labels"]
                instance_ids.extend(e["instance_ids"])
                indexed_choices.extend(e["indexed_choices"])
                choices.extend(e["choices"])
                labels_mask.append(torch.Tensor(e["labels_mask"]).long())
                # aux = []
                for lset in e_labels:
                    if len(lset) > 1:
                        idx = random.randint(0, len(lset) - 1)
                        labels.append(list(lset)[idx])
                    else:
                        labels.append(list(lset)[0])
                # indexed_labels.append(torch.Tensor(aux).long())
                # labels.append(e_labels)
            labels = torch.Tensor(labels).long()
            batched_img_features = torch.stack(img_features, 0)
            batched_img_pos = torch.stack(img_pos, 0)
            batched_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
            labels_mask = pad_sequence(labels_mask, batch_first=True, padding_value=0).bool()
            encoder_mask = batched_input_ids != self.pad_token_id
            token_type_ids = torch.zeros(batched_input_ids.shape).long()

            return {"input_ids": batched_input_ids,
                    "text_attention_mask": encoder_mask,
                    "token_type_ids": token_type_ids,
                    "visual_feats": batched_img_features,
                    "visual_pos": batched_img_pos,
                    "visual_attention_mask": torch.ones(batched_img_features.shape[:-1]),
                    "indexed_choices": indexed_choices,
                    "choices": choices,
                    "labels_mask": labels_mask,
                    "labels": labels,
                    "instance_ids": instance_ids
                    }

        return collate_fn


if __name__ == "__main__":
    # semcor_path = "/home/tommaso/Documents/data/WSD_Evaluation_Framework/Training_Corpora/SemCor/semcor.data.xml"
    # wordnet_sense_index_path = "/opt/WordNet-3.0/dict/index.sense"
    # tokid2imgid_path = "data2/training_data/multimodal/tok2image.semcor.restricted.txt"
    # imgfeat_path = "data2/training_data/multimodal/img.gcc.semcor.new.npz"
    # dataset = MultimodalTxtDataset(AutoTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased", use_fast=True),
    #                                semcor_path, tokid2imgid_path, imgfeat_path, wordnet_sense_index_path)
    #
    # data_loader = DataLoader(dataset, batch_size=4, collate_fn=dataset.get_batch_fun())
    # for batch in data_loader:
    #     pprint(batch)
    #     break
    encoder_name = "unc-nlp/lxmert-base-uncased"

    encoder_tokenizer = LxmertTokenizer.from_pretrained(encoder_name)

    semcor_path = "/home/tommaso/Documents/data/WSD_Evaluation_Framework/Training_Corpora/SemCor/semcor.data.xml"
    dev_path = "/home/tommaso/Documents/data/WSD_Evaluation_Framework/Evaluation_Datasets/semeval2007/semeval2007.data.xml"
    wordnet_sense_index_path = "/opt/WordNet-3.0/dict/index.sense"

    # train_tokid2imgid_path = "data2/training_data/multimodal/tok2image.semcor.restricted.txt"
    train_tokid2imgid_path = "/home/tommaso/dev/PycharmProjects/multimodal_wsd_bianca/data/in/wsd/semcor_sentence_image_map.txt"
    dev_tokid2imgid_path = "/home/tommaso/dev/PycharmProjects/multimodal_wsd_bianca/data/in/wsd/semeval2007_sentence_image_map.txt"

    # imgfeat_path = "data2/training_data/multimodal/img.gcc.semcor.new.npz"
    imgfeat_path = "/home/tommaso/dev/PycharmProjects/multimodal_wsd_bianca/data/in/wsd/wsd_gcc_images.npz"
    # dev_tokid2imgid_path = "data2/training_data/multimodal/tok2image.semeval2007.restricted.txt"
    # dev_imgfeat_path = "data2/training_data/multimodal/img.gcc.semeval2007.other.npz"

    img_features_files = np.load(imgfeat_path)
    logger.info("loading image features")
    img_features = img_features_files["features"]
    logger.info("image features loaded")
    all_img_boxes = img_features_files["normalized_boxes"]

    dataset = MultimodalTxtDataset(encoder_tokenizer,
                                   semcor_path, train_tokid2imgid_path, img_features_files, img_features, all_img_boxes,
                                   wordnet_sense_index_path)

    dev_dataset = MultimodalTxtDataset(encoder_tokenizer,
                                       dev_path, dev_tokid2imgid_path, img_features_files, img_features, all_img_boxes,
                                       wordnet_sense_index_path)
