import _pickle as pkl
import os
import socket
from argparse import ArgumentParser

import torch
import wandb
import yaml
from allennlp.data.iterators import BucketIterator, BasicIterator
from allennlp.data.token_indexers import PretrainedBertIndexer
from allennlp.training.callbacks import Checkpoint
from allennlp.training.checkpointer import Checkpointer
from allennlp_mods.callback_trainer import MyCallbackTrainer
from allennlp_mods.callbacks import ValidateAndWrite, WanDBTrainingCallback
from allennlp_mods.checkpointer import MyCheckpoint
from torch import optim

from src.data.dataset_utils import get_pos_from_key

from src.data.datasets import Vocabulary, AllenWSDDatasetReader
from src.evaluation.evaluate_model import evaluate_datasets
from src.misc.wsdlogging import get_info_logger
from src.models.core import PretrainedXLMIndexer, PretrainedRoBERTaIndexer
from src.models.neural_wsd_models import AllenWSDModel, WSDOutputWriter
import numpy as np

from src.utils.utils import get_token_indexer

torch.random.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

logger = get_info_logger(__name__)


def build_outpath_subdirs(path):
    if not os.path.exists(path):
        os.mkdir(path)
    try:
        os.mkdir(os.path.join(path, "checkpoints"))
    except:
        pass
    try:
        os.mkdir(os.path.join(path, "predictions"))
    except:
        pass


def main(args):
    with open(args.config) as reader:
        config = yaml.load(reader, Loader=yaml.FullLoader)
    data_config = config["data"]
    model_config = config["model"]
    training_config = config["training"]
    outpath = data_config["outpath"]
    test_data_root = data_config["test_data_root"]
    train_data_root = data_config["train_data_root"]
    test_names = data_config["test_names"]
    langs = data_config["langs"]
    sense_inventory = data_config["sense_inventory"]
    gold_id_separator = data_config["gold_id_separator"]
    label_from_training = data_config["label_from_training"]
    max_sentence_token = data_config["max_sentence_token"]
    max_segments_in_batch = data_config["max_segments_in_batch"]
    dev_name = data_config.get("dev_name", None)
    mfs_file = data_config.get("mfs_file", None)
    sliding_window = data_config["sliding_window"]
    device = model_config["device"]
    model_name = model_config["model_name"]
    learning_rate = float(model_config["learning_rate"])
    cache_instances = training_config["cache_instances"]
    num_epochs = training_config["num_epochs"]
    wandb.init(config=config, project="wsd_framework", tags=[socket.gethostname(), model_name, ",".join(langs)])
    if dev_name is None:
        logger.warning("No dev name set... In this way I won't save in best.th the best model according to the "
                       "development set. best.th will contain the weights of the model at its last epoch")
    device_int = 0 if device == "cuda" else -1
    test_paths = [os.path.join(test_data_root, name, name + ".data.xml") for name in test_names]
    training_paths = train_data_root  # "{}/SemCor/semcor.data.xml".format(train_data_root)
    outpath = os.path.join(outpath, model_name)
    build_outpath_subdirs(outpath)

    token_indexer, padding = get_token_indexer(model_name)

    if label_from_training:
        dataset_builder = AllenWSDDatasetReader.get_dataset_with_labels_from_data
    elif sense_inventory == "wnoffsets":
        dataset_builder = AllenWSDDatasetReader.get_wnoffsets_dataset
    elif sense_inventory == "sensekeys":
        dataset_builder = AllenWSDDatasetReader.get_sensekey_dataset
    elif sense_inventory == "bnoffsets":
        dataset_builder = AllenWSDDatasetReader.get_bnoffsets_dataset
    else:
        raise RuntimeError(
            "%s sense_inventory has not been recognised, ensure it is one of the following: {wnoffsets, sensekeys, bnoffsets}" % (
                sense_inventory))

    print("loading dataset")
    reader, lemma2synsets, label_vocab, mfs_dictionary = dataset_builder({"tokens": token_indexer},
                                                                         sliding_window=sliding_window,
                                                                         max_sentence_token=max_sentence_token,
                                                                         gold_id_separator=gold_id_separator,
                                                                         langs=langs,
                                                                         training_data_xmls=training_paths,
                                                                         mfs_file=mfs_file,
                                                                         sense_inventory=sense_inventory,
                                                                         lazy=data_config.get("lazy", False))
    model = AllenWSDModel.get_transformer_based_wsd_model(model_name, len(label_vocab), lemma2synsets, device_int,
                                                          label_vocab,
                                                          vocab=Vocabulary(), mfs_dictionary=mfs_dictionary,
                                                          cache_vectors=cache_instances, pad_token_id=padding)
    logger.info("loading training data...")
    train_ds = reader.read(training_paths)

    #####################################################
    # NEDED so to not split sentences in the test data. #
    reader.max_sentence_len = 200
    reader.sliding_window_size = 200
    #####################################################
    logger.info("loading test data...")
    tests_dss = [reader.read(test_path) for test_path in test_paths]
    # iterator = BasicIterator(maximum_samples_per_batch=("tokens_length", max_segments_in_batch),
    #                          cache_instances=True
    #                          )
    iterator = BucketIterator(
        biggest_batch_first=True,
        sorting_keys=[("tokens", "num_tokens")],
        maximum_samples_per_batch=("tokens_length", max_segments_in_batch),
        cache_instances=True,
        
        #instances_per_epoch=10
    )
    valid_iterator = BucketIterator(
        maximum_samples_per_batch=("tokens_length", max_segments_in_batch),
        biggest_batch_first=True,
        sorting_keys=[("tokens", "num_tokens")],
        cache_instances=True
        # instances_per_epoch=10

    )
    iterator.index_with(Vocabulary())
    writers = [WSDOutputWriter(os.path.join(outpath, "predictions", name + ".predictions.txt"), label_vocab.itos) for
               name
               in test_names]
    callbacks = [ValidateAndWrite(data, valid_iterator, output_writer=writer, name=name, wandb=True,
                                  is_dev=name == dev_name if dev_name is not None else False) for
                 name, data, writer in zip(
            test_names, tests_dss, writers)]
    callbacks.append(WanDBTrainingCallback())
    callbacks.append(
        MyCheckpoint(Checkpointer(os.path.join(outpath, "checkpoints"), num_serialized_models_to_keep=100),
                     autoload_last_checkpoint=args.reload_checkpoint))

    trainer = MyCallbackTrainer(model=model,
                                optimizer=optim.Adam(model.parameters(), lr=learning_rate),
                                iterator=iterator,
                                cuda_device=device_int,
                                num_epochs=num_epochs,
                                training_data=train_ds,
                                callbacks=callbacks,
                                shuffle=True,
                                track_dev_metrics=True,
                                metric_name="f1_mfs" if mfs_file else "f1"
                                )
    trainer.train()
    with open(os.path.join(outpath, "last_model.th"), "wb") as writer:
        torch.save(model.state_dict(), writer)
    with open(os.path.join(outpath, "label_vocab.pkl"), "wb") as writer:
        pkl.dump(label_vocab, writer)
    if not os.path.exists(os.path.join(outpath, "evaluation")):
        os.mkdir(os.path.join(outpath, "evaluation"))
    evaluate_datasets(test_paths, reader, os.path.join(outpath, "checkpoints", "best.th"), model_name, label_vocab,
                      lemma2synsets, device_int,
                      mfs_dictionary, mfs_dictionary is not None, os.path.join(outpath, "evaluation"), padding,
                      verbose=True,
                      debug=False)


# os.environ["WANDB_MODE"] = "dryrun"
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", required=True)  # default="config/config_es_s+g+o.yaml")
    parser.add_argument("--dryrun", action="store_true")
    parser.add_argument("--reload_checkpoint", action="store_true", default=False)
    args = parser.parse_args()
    if args.dryrun:
        os.environ["WANDB_MODE"] = "dryrun"
    print("config {}".format(args.config))
    main(args)
