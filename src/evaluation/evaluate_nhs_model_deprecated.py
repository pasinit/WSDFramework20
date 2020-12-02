import os
import subprocess
from argparse import ArgumentParser
from collections import OrderedDict
from typing import Dict, List

import torch
import yaml
from allennlp.data import Vocabulary
from allennlp.data.iterators import BasicIterator
from allennlp.predictors import SentenceTaggerPredictor
from data_io.datasets import AllenWSDDatasetReader
from pandas import DataFrame
from tqdm import tqdm

from src.data.dataset_utils import get_dataset_with_labels_from_data, get_wnoffsets_dataset, get_sensekey_dataset, \
    get_bnoffsets_dataset, get_label_mapper
from src.models.neural_wsd_models import AllenWSDModel, WSDF1
# from src.training.wsd_trainer import get_token_indexer
from src.utils.utils import get_token_indexer


def evaluate(dataset_reader, dataset_path, model, output_path, label_vocab, use_mfs=False, mfs_vocab=None,
             verbose=True, debug=False):
    iterator = BasicIterator(
        batch_size=64
    )
    # iterator = BucketIterator(
    #     biggest_batch_first=False,
    #     sorting_keys=[("tokens", "num_tokens")],
    #     maximum_samples_per_batch=("tokens_length", max_segments_in_batch),
    # )
    f1_computer = WSDF1(label_vocab, use_mfs, mfs_vocab)
    predictor = SentenceTaggerPredictor(model, dataset_reader)
    batches = [x for x in iterator._create_batches(dataset_reader.read(dataset_path, label_mapper_getter=get_label_mapper), False)]
    with open(output_path, "w") as writer, open(output_path.replace(".txt", ".mfs.txt"), "w") as mfs_writer, \
            open(output_path.replace(".txt", ".mfs.info.txt"), "w") as mfs_info_writer:
        bar = batches
        if verbose:
            bar = tqdm(batches)
        debugwriter = open("/tmp/debug.txt", "w")
        for batch in bar:
            outputs = predictor.predict_batch_instance(batch.instances)
            ids = [prediction["ids"] for prediction in outputs]
            predictions = [prediction["full_predictions"] for prediction in outputs]
            for i_ids, i_predictions, instance in zip(ids, predictions, batch):
                i_predictions = [int(x) for x in i_predictions if x > 0.0]
                i_labels = instance.fields["labels"].metadata
                assert len(i_ids) == len(i_predictions)
                lemmapos = instance.fields["labeled_lemmapos"]
                f1_computer(lemmapos, i_predictions, i_labels, ids=instance.fields["ids"].metadata)
                f1_computer.get_metric(False)
                writer.write(
                    "\n".join(["{} {}".format(id, label_vocab.itos[p]) for id, p in zip(i_ids, i_predictions)]))
                writer.write("\n")
                if mfs_vocab is not None:
                    mfs_preds = [mfs_vocab.get(lp, "<unk>") if label_vocab.itos[p] == "<unk>" else label_vocab.itos[p]
                                 for
                                 lp, p in
                                 zip(lemmapos, i_predictions)]
                    mfs_writer.write(
                        "\n".join(["{} {}".format(id, p) for id, p in zip(i_ids, mfs_preds)]))
                    mfs_writer.write("\n")
                if debug:
                    debugwriter.write(" ".join(x.text for x in instance.fields["tokens"]) + "\n")
                    for id, lp, p, label, possible_labels in zip(i_ids, lemmapos, i_predictions, i_labels,
                                                                 [[label_vocab.get_string(y) for y in x] for x in
                                                                  instance.fields["possible_labels"]]):
                        is_mfs = label_vocab.itos[p] == "<unk>"
                        pred = mfs_vocab.get(lp, "<unk>") if label_vocab.itos[p] == "<unk>" else label_vocab.itos[p]
                        mfs_info_writer.write("{} {} {}\n".format(id, pred, "MFS" if is_mfs else ""))
                        debugwriter.write(
                            "{}\t{}\t{}\t{}\t{}\n".format(id, lp, pred, label, ", ".join(possible_labels)))
        metric = f1_computer.get_metric(True)
        debugwriter.close()
        return metric


import pandas


def evaluate_datasets(dataset_paths: List[str],
                      dataset_reader: AllenWSDDatasetReader, checkpoint_path: str, model_name: str, label_vocab: Dict,
                      lemma2synsets: Dict,
                      device_int: int, mfs_dictionary: Dict,
                      use_mfs: bool,
                      output_path,
                      padding,
                      verbose=True, debug=False,

                      ):
    model = AllenWSDModel.get_transformer_based_wsd_model(model_name, len(label_vocab), lemma2synsets, device_int,
                                                          label_vocab,
                                                          vocab=Vocabulary(), mfs_dictionary=mfs_dictionary,
                                                          eval=True, cache_instances=False,
                                                          pad_token_id=padding, return_full_output=True)
    model.load_state_dict(
        torch.load(checkpoint_path, map_location="cpu" if device_int < 0 else "cuda:{}".format(device_int)))
    model.eval()
    all_metrics = dict()
    names = list()  # [dataset_path.split("/")[-1].split(".")[0] for dataset_path in dataset_paths]
    lines = list()
    if not verbose:
        dataset_paths = tqdm(dataset_paths, desc="datasets_progress")
    for dataset_path in dataset_paths:
        name = dataset_path.split("/")[-1]  # .split(".")[0]
        names.append(name)
        metrics = evaluate(dataset_reader, dataset_path, model, os.path.join(output_path, name + ".predictions.txt"),
                           label_vocab, use_mfs, mfs_dictionary, verbose=verbose, debug=debug)
        all_metrics[name] = OrderedDict(
            {"precision": metrics["precision"], "recall": metrics["recall"], "f1": metrics["f1"],
             "f1_mfs": metrics.get("f1_mfs", None)})
        if verbose:
            print("{}: precision: {}, recall: {}, f1: {}, precision_mfs: {}, recall_mfs: {}, f1_mfs:{}".format(name,
                                                                                                               *[
                                                                                                                   metrics.get(
                                                                                                                       x,
                                                                                                                       -1)
                                                                                                                   for x
                                                                                                                   in
                                                                                                                   [
                                                                                                                       "precision",
                                                                                                                       "recall",
                                                                                                                       "f1",
                                                                                                                       "p_mfs",
                                                                                                                       "recall_mfs",
                                                                                                                       "f1_mfs"]]))
        lines.append([metrics["precision"], metrics["recall"], metrics["f1"], metrics.get("f1_mfs", -1)])
    print("SUMMARY:")
    # d = DataFrame.from_dict(all_metrics).transpose()
    d = DataFrame(lines, columns=["Precision", "Recall", "F1", "F1_MFS"], index=names)
    with pandas.option_context('display.max_rows', None, 'display.max_columns', None, 'display.float_format',
                               '{:0.3f}'.format):  # more options can be specified also
        print(d.to_csv(sep="\t"))


def to_scorer_format(path):
    outpath = ".tmp_eval_preds.txt"
    with open(path) as lines, open(outpath, "w") as writer:
        for line in lines:
            writer.write(" ".join(line.strip().split("\t")[:-1]))


def get_best_checkpoint_from_predictions(dataset_path, dataset_reader, prediction_dir, model_name, label_vocab,
                                         lemma2synsets, device_int,
                                         mfs_dictionary, use_mfs):
    for i in range(len([x for x in os.listdir(prediction_dir) if "semeval2007.predictions" in x]) - 1):
        print("Epoch: {}".format(i))
        prediction_path = os.path.join(prediction_dir,
                                       "semeval2007.predictions.txt{}".format(".{}".format(i) if i > 0 else ""))
        tmp_path = to_scorer_format(prediction_path)
        java_run = "-cp Documents/data/WSD_Evaluation_Framework/Evaluation_Datasets/ Scorer " \
                   "{} {}".format(dataset_path, tmp_path)
        subprocess.call('java' + java_run, shell=True)


def main(args):
    with open(args.config) as reader:
        config = yaml.load(reader, Loader=yaml.FullLoader)
    verbose = args.verbose
    debug = args.debug
    data_config = config["data"]
    model_config = config["model"]
    outpath = data_config["outpath"]
    test_data_root = data_config["test_data_root"]
    train_data_root = data_config["train_data_root"]
    test_names = data_config["test_names"]
    langs = data_config["langs"]
    sense_inventory = data_config["sense_inventory"]
    gold_id_separator = data_config["gold_id_separator"]
    label_from_training = data_config["label_from_training"]
    # max_sentence_token = data_config["max_sentence_token"]
    # max_segments_in_batch = data_config["max_segments_in_batch"]
    mfs_file = data_config.get("mfs_file", None)
    # sliding_window = data_config["sliding_window"]
    device = model_config["device"]
    model_name = model_config["model_name"]
    checkpoint_path = args.checkpoint_path
    device_int = 0 if device == "cuda" else -1
    if args.test_path is None:
        test_paths = [os.path.join(test_data_root, name, name + ".data.xml") for name in test_names]
    else:
        if len(args.test_path) == 1 and os.path.isdir(args.test_path[0]):
            test_paths = [os.path.join(args.test_path[0], p) for p in os.listdir(args.test_path[0]) if "data.xml" in p]
        else:
            test_paths = args.test_path
    training_paths = train_data_root  # "{}/SemCor/semcor.data.xml".format(train_data_root)
    # outpath = os.path.join(outpath, model_name)
    token_indexer, padding = get_token_indexer(model_name)

    if label_from_training:
        dataset_builder = get_dataset_with_labels_from_data
    elif sense_inventory == "wnoffsets":
        dataset_builder = get_wnoffsets_dataset
    elif sense_inventory == "sensekeys":
        dataset_builder = get_sensekey_dataset
    elif sense_inventory == "bnoffsets":
        dataset_builder = get_bnoffsets_dataset
    else:
        raise RuntimeError(
            "%s sense_inventory has not been recognised, ensure it is one of the following: {wnoffsets, sensekeys, bnoffsets}" % (
                sense_inventory))

    reader, lemma2synsets, label_vocab, mfs_dictionary = dataset_builder({"tokens": token_indexer},
                                                                         sliding_window=35,
                                                                         max_sentence_token=35,
                                                                         gold_id_separator=gold_id_separator,
                                                                         langs=langs,
                                                                         training_data_xmls=training_paths,
                                                                         sense_inventory=sense_inventory,
                                                                         mfs_file=mfs_file, )
    if args.find_best is True:
        epoch = get_best_checkpoint(args.dev_set, reader, checkpoint_path, model_name,
                                    label_vocab, lemma2synsets, device_int,
                                    mfs_dictionary, args.output_path, mfs_dictionary is not None,
                                    verbose=verbose
                                    )
        checkpoint_path = os.path.join(checkpoint_path, "model_state_epoch_{}.th".format(epoch))
        print("best checpoint: {}".format(checkpoint_path))
    evaluate_datasets(test_paths, reader, checkpoint_path, model_name, label_vocab, lemma2synsets, device_int,
                      mfs_dictionary, mfs_dictionary is not None, args.output_path, padding, verbose=verbose,
                      debug=debug)


def get_best_checkpoint(path, reader, checkpoint_path, model_name, label_vocab, lemma2synsets, device_int,
                        mfs_dictionary, output_path, use_mfs, metric_to_track="f1_mfs", verbose=True):
    model = AllenWSDModel.get_transformer_based_wsd_model(model_name, len(label_vocab), lemma2synsets, device_int,
                                                          label_vocab,
                                                          vocab=Vocabulary(), mfs_dictionary=mfs_dictionary,
                                                          eval=True, finetune_embedder=False, return_full_output=True)
    num_checkpoints = len([x for x in os.listdir(checkpoint_path) if "model_state_epoch" in x])
    best_epoch = -1
    best_metric = -1
    r = range(num_checkpoints)
    if not verbose:
        r = tqdm(r)
    for epoch in r:
        fname = "model_state_epoch_{}.th".format(epoch)
        model.load_state_dict(
            torch.load(os.path.join(checkpoint_path, fname),
                       map_location="cpu" if device_int < 0 else "cuda:{}".format(device_int)))
        model.eval()
        name = path.split("/")[-1]
        metrics = evaluate(reader, path, model,
                           os.path.join(output_path, name + ".predictions.txt"),
                           label_vocab, use_mfs, mfs_dictionary, verbose=verbose)
        val = metrics[metric_to_track]
        if val > best_metric:
            best_epoch = epoch
            best_metric = val
            print("new best: epoch {}, {}: {}".format(epoch, metric_to_track, val))
    return best_epoch


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--test_path", default=None, nargs="+")
    parser.add_argument("--find_best", action="store_true", default=False)
    parser.add_argument("--dev_set", default=None, type=str)
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)

    args = parser.parse_args()
    main(args)
