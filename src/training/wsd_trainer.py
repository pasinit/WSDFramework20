import _pickle as pkl
import os
import socket
from argparse import ArgumentParser

import torch
import wandb
import yaml
from allennlp.data.iterators import BucketIterator
from allennlp.data.token_indexers import PretrainedBertIndexer
from allennlp.training.callbacks import Checkpoint
from allennlp.training.checkpointer import Checkpointer
from allennlp_mods.callback_trainer import MyCallbackTrainer
from allennlp_mods.callbacks import ValidateAndWrite, WanDBTrainingCallback
from allennlp_mods.checkpointer import MyCheckpoint
from torch import optim

from src.data.dataset_utils import get_pos_from_key

from src.data.datasets import Vocabulary, AllenWSDDatasetReader
from src.misc.logging import get_info_logger
from src.models.neural_wsd_models import AllenWSDModel, WSDOutputWriter
import numpy as np

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
    label_from = data_config["label_from"]
    max_sentence_token = data_config["max_sentence_token"]
    max_segments_in_batch = data_config["max_segments_in_batch"]
    mfs_file = data_config.get("mfs_file", None)
    sliding_window = data_config["sliding_window"]
    device = model_config["device"]
    model_name = model_config["model_name"]
    learning_rate = float(model_config["learning_rate"])
    num_epochs = training_config["num_epochs"]
    wandb.init(config=config, project="wsd_framework", tags=[socket.gethostname(), model_name, ",".join(langs)])
    device_int = 0 if device == "cuda" else -1
    test_paths = [os.path.join(test_data_root, name, name + ".data.xml") for name in test_names]
    training_paths = train_data_root  # "{}/SemCor/semcor.data.xml".format(train_data_root)
    outpath = os.path.join(outpath, model_name)
    build_outpath_subdirs(outpath)
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
                                                                         sliding_window=sliding_window,
                                                                         max_sentence_token=max_sentence_token,
                                                                         gold_id_separator=gold_id_separator,
                                                                         langs=langs,
                                                                         training_data_xmls=training_paths,
                                                                         sense_inventory=sense_inventory,
                                                                         mfs_file=mfs_file)
    model = AllenWSDModel.get_bert_based_wsd_model(model_name, len(label_vocab), lemma2synsets, device_int, label_vocab,
                                                   vocab=Vocabulary(), mfs_dictionary=mfs_dictionary,
                                                   cache_vectors=True)
    logger.info("loading training data...")
    train_ds = reader.read(training_paths)
    #####################################################
    # NEDED so to not split sentences in the test data. #
    reader.max_sentence_len = 200
    reader.sliding_window_size = 200
    #####################################################
    logger.info("loading test data...")
    tests_dss = [reader.read(test_path) for test_path in test_paths]
    iterator = BucketIterator(
        biggest_batch_first=True,
        sorting_keys=[("tokens", "num_tokens")],
        maximum_samples_per_batch=("tokens_length", max_segments_in_batch),
        cache_instances=True,
<<<<<<< HEAD
        #instances_per_epoch=10
=======
>>>>>>> aeaa23277e2054e5a74fb53b17a2e4864b231581
    )
    valid_iterator = BucketIterator(
        maximum_samples_per_batch=("tokens_length", max_segments_in_batch),
        biggest_batch_first=True,
        sorting_keys=[("tokens", "num_tokens")],
        # instances_per_epoch=10

    )
    iterator.index_with(Vocabulary())
    writers = [WSDOutputWriter(os.path.join(outpath, "predictions", name + ".predictions.txt"), label_vocab.itos) for
               name
               in test_names]
    callbacks = [ValidateAndWrite(data, valid_iterator, output_writer=writer, name=name, wandb=True, is_dev=name=="semeval2007") for
                 name, data, writer in zip(
            test_names, tests_dss, writers)]
    callbacks.append(WanDBTrainingCallback())
    callbacks.append(
        MyCheckpoint(Checkpointer(os.path.join(outpath, "checkpoints"), num_serialized_models_to_keep=100)))

    trainer = MyCallbackTrainer(model=model,
                                optimizer=optim.Adam(model.parameters(), lr=learning_rate),
                                iterator=iterator,
                                cuda_device=device_int,
                                num_epochs=num_epochs,
                                training_data=train_ds,
                                callbacks=callbacks,
                                shuffle=True,
                                track_dev_metrics=True,
                                metric_name="f1_mfs"
                                )
    trainer.train()
    with open(os.path.join(outpath, "last_model.th"), "wb") as writer:
        torch.save(model.state_dict(), writer)
    with open(os.path.join(outpath, "label_vocab.pkl"), "wb") as writer:
        pkl.dump(label_vocab, writer)


os.environ["WANDB_MODE"] = "dryrun"
if __name__ == "__main__":
    parser = ArgumentParser()
<<<<<<< HEAD
    parser.add_argument("--config", required=True)#default="config/config_es_s+g+o.yaml")
=======
    parser.add_argument("--config", default="config/config_en_semcor_sensekey.yaml")
>>>>>>> aeaa23277e2054e5a74fb53b17a2e4864b231581
    parser.add_argument("--dryrun", action="store_true")
    args = parser.parse_args()
    if args.dryrun:
        os.environ["WANDB_MODE"] = "dryrun"
    print("config {}".format(args.config))
    main(args)
