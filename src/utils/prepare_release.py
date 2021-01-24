from typing import List
from src.data.dataset_utils import get_allen_datasets, get_data, get_mapper
import os
from tqdm import tqdm


def write_datasets_bn(training_paths: List, outpath: str, langs=("en",)):
    inventory = "bnoffsets"
    sense_inventory = inventory
    lemma2synsets, _, label_vocab = get_data(
        inventory, langs, None, invetory_dir=None)
    label_mapper = get_mapper(training_paths, sense_inventory)
    training_ds, training_iterator = get_allen_datasets(None, "xlm-roberta-base",
                                                        lemma2synsets, label_vocab, label_mapper,
                                                        1000, training_paths, True, False, True)
    with open(outpath, "w") as writer:
        for example in tqdm(training_ds):
            for id, labels in [elem for elem in zip(example["ids"], example["labels"]) if elem[0] is not None]:
                writer.write(id + " " + " ".join(labels) + "\n")


def convert_training_data(output_dir: str):
    training_paths = {
        "en": ["data2/training_data/en_training_data/semcor/semcor.data.xml"]}
    langs = ("en",)
    write_datasets_bn(training_paths, os.path.join(
        output_dir, "semcor.gold.key.txt"), langs)
    training_paths = {"en": [
        "data2/training_data/en_training_data/wngt_michele/wngt_michele_examples/wngt_michele_examples.data.xml"]}
    write_datasets_bn(training_paths, os.path.join(
        output_dir, "wngt_examples.gold.key.txt"), langs)

    training_paths = {"en": [
        "data2/training_data/en_training_data/wngt_michele/wngt_michele_glosses/wngt_michele_glosses.data.xml"]}
    write_datasets_bn(training_paths, os.path.join(
        output_dir, "wngt_glosses.gold.key.txt"), langs)

def convert_en_test_data(output_dir: str):
    langs = ("en",)
    paths = {"en": ["resources/evaluation_framework_3.0/new_evaluation_datasets/test-en/test-en.data.xml"]}
    write_datasets_bn(paths, os.path.join(output_dir, "test-en.gold.key.txt"), langs)

    paths = {"en": ["resources/evaluation_framework_3.0/new_evaluation_datasets/test-en-coarse/test-en-coarse.data.xml"]}
    write_datasets_bn(paths, os.path.join(output_dir, "test-en-coarse.gold.key.txt"), langs)
    
    paths = {"en": ["resources/evaluation_framework_3.0/new_evaluation_datasets/dev-en/dev-en.data.xml"]}
    write_datasets_bn(paths, os.path.join(output_dir, "dev-en.gold.key.txt"), langs)
    
if __name__ == "__main__":
    outdir = "/tmp/dataset_test/"
    # convert_training_data(outdir)
    convert_en_test_data(outdir)
