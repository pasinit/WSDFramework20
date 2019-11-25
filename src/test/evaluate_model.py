from typing import Dict, List, OrderedDict

from allennlp.data import Vocabulary
from allennlp.data.iterators import BucketIterator, BasicIterator
from allennlp.data.token_indexers import PretrainedBertIndexer
from allennlp.predictors import SentenceTaggerPredictor
from allennlp.tests.commands.evaluate_test import DummyIterator
from argparse import ArgumentParser

from pandas import DataFrame
from torchtext.data import BucketIterator
from tqdm import tqdm

from src.data.datasets import AllenWSDDatasetReader
from src.models.neural_wsd_models import AllenWSDModel, WSDF1
import torch

import os
import yaml

def evaluate(dataset_reader, dataset_path, model, output_path, label_vocab, use_mfs=False, mfs_vocab=None):
    iterator = BasicIterator(
    )
    f1_computer = WSDF1(label_vocab, use_mfs, mfs_vocab)
    predictor = SentenceTaggerPredictor(model, dataset_reader)
    batches = [x for x in iterator._create_batches(dataset_reader.read(dataset_path), False)]
    with open(output_path, "w") as writer, open(output_path.replace(".txt", ".mfs.txt"), "w") as mfs_writer:
        for batch in tqdm(batches):
            outputs = predictor.predict_batch_instance(batch.instances)
            ids = [prediction["ids"] for prediction in outputs]
            predictions = [prediction["full_predictions"] for prediction in outputs]
            for i_ids, i_predictions, instance in zip(ids, predictions, batch):
                i_predictions = [int(x) for x in i_predictions if x > 0.0]
                i_labels = instance.fields["labels"].metadata
                assert len(i_ids) == len(i_predictions)
                lemmapos = instance.fields["labeled_lemmapos"]
                f1_computer(lemmapos, i_predictions, i_labels, ids=instance.fields["ids"].metadata)
                writer.write("\n".join(["{} {}".format(id, label_vocab.itos[p]) for id, p in zip(i_ids, i_predictions)]))
                writer.write("\n")
                mfs_preds = [mfs_vocab[lp] if label_vocab.itos[p] == "<unk>" else label_vocab.itos[p] for lp, p in zip(lemmapos, i_predictions)]
                mfs_writer.write(
                    "\n".join(["{} {}".format(id, p) for id, p in zip(i_ids, mfs_preds)]))
                mfs_writer.write("\n")

        metric = f1_computer.get_metric(True)
        return metric


import pandas
def evaluate_datasets(dataset_paths: List[str],
                      dataset_reader: AllenWSDDatasetReader, checkpoint_path: str, model_name: str, label_vocab: Dict,
                      lemma2synsets: Dict,
                      device_int: int, mfs_dictionary: Dict,
                      use_mfs:bool,
                      output_path
                      ):
    model = AllenWSDModel.get_bert_based_wsd_model(model_name, len(label_vocab), lemma2synsets, device_int, label_vocab,
                                                   vocab=Vocabulary(), mfs_dictionary=mfs_dictionary,
                                                   cache_vectors=True, return_full_output=True)
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    all_metrics = dict()
    names = list()#[dataset_path.split("/")[-1].split(".")[0] for dataset_path in dataset_paths]
    lines = list()
    for dataset_path in dataset_paths:
        name = dataset_path.split("/")[-1].split(".")[0]
        names.append(name)
        metrics = evaluate(dataset_reader, dataset_path, model, os.path.join(output_path, name + ".predictions.txt"), label_vocab, use_mfs, mfs_dictionary)

        all_metrics[name] =  OrderedDict({"precision": metrics["precision"], "recall": metrics["recall"], "f1": metrics["f1"],
                              "f1_mfs": metrics["f1_mfs"]})
        print("{}: precision: {}, recall: {}, f1: {}, precision_mfs: {}, recall_mfs: {}, f1_mfs:{}".format(name, *[metrics[x]
                                                                                                                   for x in ["precision", "recall", "f1", "p_mfs", "recall_mfs", "f1_mfs"]]))
        lines.append([metrics["precision"], metrics["recall"], metrics["f1"], metrics["f1_mfs"]])
        break
    print("SUMMARY:")
    # d = DataFrame.from_dict(all_metrics).transpose()
    d = DataFrame(lines, columns=["Precision", "Recall", "F1", "F1_MFS"], index=names)
    with pandas.option_context('display.max_rows', None, 'display.max_columns', None, 'display.float_format', '{:0.3f}'.format):  # more options can be specified also
        print(d.to_csv(sep="\t"))
    # for name, metrics in all_metrics.items():

    # for batch in iterator(dataset, num_epochs=1, shuffle=False):
    #     predictor.




def main(args):
    with open(args.config) as reader:
        config = yaml.load(reader, Loader=yaml.FullLoader)
    data_config = config["data"]
    model_config = config["model"]
    outpath = data_config["outpath"]
    test_data_root = data_config["test_data_root"]
    train_data_root = data_config["train_data_root"]
    test_names = data_config["test_names"]
    langs = data_config["langs"]
    sense_inventory = data_config["sense_inventory"]
    gold_id_separator = data_config["gold_id_separator"]
    label_from = data_config["label_from"]
    max_sentence_token = data_config["max_sentence_token"]
    max_segments_in_batch = data_config["max_segments_in_batch"]
    mfs_file = data_config.get("mfs_file", None)
    sliding_window = data_config["sliding_window"]
    device = model_config["device"]
    model_name = model_config["model_name"]
    checkpoint_path = args.checkpoint_path
    device_int = 0 if device == "cuda" else -1
    test_paths = [os.path.join(test_data_root, name, name + ".data.xml") for name in test_names]
    training_paths = train_data_root  # "{}/SemCor/semcor.data.xml".format(train_data_root)
    outpath = os.path.join(outpath, model_name)
    token_indexer = PretrainedBertIndexer(
        pretrained_model=model_name,
        do_lowercase=False,
        truncate_long_sequences=False
    )

    if label_from == "wnoffsets":
        dataset_builder = AllenWSDDatasetReader.get_wnoffsets_dataset
    elif label_from == "sensekeys":
        dataset_builder = AllenWSDDatasetReader.get_sensekey_dataset
    elif label_from == "bnids":
        dataset_builder = AllenWSDDatasetReader.get_bnoffsets_dataset
    elif label_from == "training":
        dataset_builder = AllenWSDDatasetReader.get_dataset_with_labels_from_data

    else:
        raise RuntimeError(
            "%s label_from has not been recognised, ensure it is one of the following: {wnoffsets, sensekeys, bnids}" % (
                label_from))

    reader, lemma2synsets, label_vocab, mfs_dictionary = dataset_builder({"tokens": token_indexer},
                                                                         sliding_window=200,
                                                                         max_sentence_token=200,
                                                                         gold_id_separator=gold_id_separator,
                                                                         langs=langs,
                                                                         training_data_xmls=training_paths,
                                                                         sense_inventory=sense_inventory,
                                                                         mfs_file=mfs_file,)
    evaluate_datasets(test_paths, reader, checkpoint_path, model_name, label_vocab, lemma2synsets, device_int,
                      mfs_dictionary, mfs_dictionary is not None, args.output_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", default="config/config_en_semcor_sensekey.yaml")
    parser.add_argument("--checkpoint_path",
                        default="data/models/en_semcor_sensekeys_mfs/bert-large-cased/checkpoints/model_state_epoch_20.th")
    parser.add_argument("--output_path", default="data/models/en_semcor_sensekeys_mfs/bert-large-cased/evaluation/")

    args = parser.parse_args()
    main(args)