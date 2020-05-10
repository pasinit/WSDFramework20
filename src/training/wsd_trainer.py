import os
import _pickle as pkl
import hashlib
import os
import socket
from argparse import ArgumentParser

import numpy as np
import torch
import wandb
import yaml
from allennlp.data import Vocabulary
from allennlp.training import GradientDescentTrainer
from nlp_tools.allen_data.iterators import get_bucket_iterator
from nlp_tools.allennlp_training_callbacks.callbacks import WanDBTrainingCallback, TestAndWrite
from torch.optim import Adam

from src.data.dataset_utils import get_wnoffsets_dataset, get_sensekey_dataset, \
    get_bnoffsets_dataset, get_label_mapper
from src.evaluation.evaluate_model import evaluate_datasets
from src.misc.wsdlogging import get_info_logger
from src.models.neural_wsd_models import WSDOutputWriter
from src.utils.utils import get_model

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


def get_dataset_builder(sense_inventory):
    if sense_inventory == "wnoffsets":
        return get_wnoffsets_dataset
    elif sense_inventory == "sensekeys":
        return get_sensekey_dataset
    elif sense_inventory == "bnoffsets":
        return get_bnoffsets_dataset
    else:
        raise RuntimeError(
            "%s sense_inventory has not been recognised, ensure it is one of the following: {wnoffsets, sensekeys, bnoffsets}" % (
                sense_inventory))


def get_mapper(training_paths, sense_inventory):
    all_labels = list()
    for f in training_paths:
        with open(f.replace(".data.xml", ".gold.key.txt")) as reader:
            all_labels.extend([l.split(" ")[1] for l in reader])
    label_mapper = get_label_mapper(target_inventory=sense_inventory, labels=all_labels)
    if len(label_mapper) > 0:  ## handles the case when training set has a key set and test sets h
        for k, v in list(label_mapper.items()):
            for x in v:
                label_mapper[x] = [x]
    return label_mapper


def get_cached_dataset_file_name(*args):
    m = hashlib.sha256()
    for arg in args:
        m.update(bytes(str(arg), 'utf8'))
    return m.hexdigest()


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
    force_reload = data_config["force_reload"]
    sense_inventory = data_config["sense_inventory"]
    max_segments_in_batch = data_config["max_segments_in_batch"]
    dev_name = data_config.get("dev_name", None)
    mfs_file = data_config.get("mfs_file", None)
    device = model_config["device"]
    encoder_name = model_config["encoder_name"]
    wsd_model_name = model_config["wsd_model_name"]
    layers_ = model_config.get("layers_to_use", (-4, -3, -2, -1))
    finetune_embedder = model_config.get("finetune", False)
    learning_rate = float(model_config["learning_rate"])
    cache_instances = training_config["cache_instances"]
    num_epochs = training_config["num_epochs"]
    wandb.init(config=config, project="wsd_framework_3.0", tags=[socket.gethostname(), encoder_name, ",".join(langs)])
    if dev_name is None:
        logger.warning("No dev name set... In this way I won't save in best.th the best model according to the "
                       "development set. best.th will contain the weights of the model at its last epoch")
    device_int = 0 if device == "cuda" else -1
    test_paths = [os.path.join(test_data_root, name, name + ".data.xml") for name in test_names]
    training_paths = train_data_root
    outpath = os.path.join(outpath, encoder_name)
    build_outpath_subdirs(outpath)

    dataset_builder = get_dataset_builder(sense_inventory)
    label_mapper = get_mapper(training_paths, sense_inventory)
    cached_dataset_file_name = get_cached_dataset_file_name(encoder_name, sense_inventory, training_paths,
                                                            max_segments_in_batch)
    label_vocab, lemma2synsets, mfs_dictionary, training_ds, training_iterator = get_training_data(
        cached_dataset_file_name, dataset_builder, encoder_name, label_mapper, langs, max_segments_in_batch, mfs_file,
        training_paths, force_reload)

    test_dss = get_test_datasets(dataset_builder, encoder_name, label_mapper, langs, mfs_file, test_paths)

    dev_ds = get_dev_dataset(dev_name, test_dss, test_names)

    model = get_model(cache_instances, device_int, encoder_name, finetune_embedder, label_vocab, layers_, lemma2synsets,
                      mfs_dictionary, model_config, training_ds, wsd_model_name)

    dev_iterator = None
    if dev_ds is not None:
        dev_iterator = get_bucket_iterator(dev_ds, max_segments_in_batch)

    writers = [WSDOutputWriter(os.path.join(outpath, "predictions", name + ".predictions.txt"), label_vocab.itos)
               for name in test_names]
    test_data_loaders = [get_bucket_iterator(td, max_segments_in_batch) for td in
                         test_dss]
    callbacks = [TestAndWrite(test_iterator=td,
                              output_writer=writer,
                              name=name,
                              wandb=True,
                              is_dev=name == dev_name if dev_name is not None else False)
                 for name, td, writer in zip(test_names, test_data_loaders, writers)]
    callbacks.append(WanDBTrainingCallback())
    trainer = GradientDescentTrainer(model=model,
                                     optimizer=Adam(model.parameters(), lr=learning_rate),
                                     data_loader=training_iterator,
                                     cuda_device=device_int,
                                     grad_clipping=1.0,
                                     num_epochs=num_epochs,
                                     validation_data_loader=dev_iterator,
                                     num_gradient_accumulation_steps=training_config.get("gradient_accumulation", 1),
                                     validation_metric="+f1",  # "+f1_mfs" if mfs_file else "+f1",
                                     # validation_metric="-loss",
                                     epoch_callbacks=callbacks,
                                     # serialization_dir=os.path.join(outpath, "checkpoints"),
                                     # checkpointer=Checkpointer(os.path.join(outpath, "checkpoints"),
                                     #                           num_serialized_models_to_keep=100),
                                     )
    trainer.train()
    with open(os.path.join(outpath, "last_model.th"), "wb") as writer:
        torch.save(model.state_dict(), writer)
    with open(os.path.join(outpath, "label_vocab.pkl"), "wb") as writer:
        pkl.dump(label_vocab, writer)
    if not os.path.exists(os.path.join(outpath, "evaluation")):
        os.mkdir(os.path.join(outpath, "evaluation"))
    evaluate_datasets(wsd_model_name,
                      layers_,
                      test_data_loaders, test_names, os.path.join(outpath, "checkpoints", "best.th"), encoder_name,
                      label_vocab,
                      lemma2synsets, device_int,
                      mfs_dictionary, mfs_dictionary is not None, os.path.join(outpath, "evaluation"),
                      training_ds.pad_token_id,
                      verbose=True,
                      debug=False)


def get_dev_dataset(dev_name, test_dss, test_names):
    dev_ds = None
    if dev_name is not None:
        dev_ds = test_dss[test_names.index(dev_name)]
        dev_ds.index_with(Vocabulary())
    return dev_ds


def get_test_datasets(dataset_builder, encoder_name, label_mapper, langs, mfs_file, test_paths):
    get_cached_dataset_file_name(*test_paths, encoder_name)
    test_dss = [dataset_builder(encoder_name, t, label_mapper, langs, mfs_file)[0] for t in test_paths]
    for td in test_dss:
        td.index_with(Vocabulary())
    return test_dss


def get_training_data(cached_dataset_file_name, dataset_builder, encoder_name, label_mapper, langs,
                      max_segments_in_batch, mfs_file, training_paths, force_reload):
    if not force_reload is not None and os.path.exists(os.path.join(".cache/", cached_dataset_file_name)):
        logger.info("Loading training set from cache: {}".format(os.path.join(".cache/", cached_dataset_file_name)))
        with open(os.path.join(".cache/", cached_dataset_file_name), "rb") as reader:
            training_iterator, training_ds, lemma2synsets, mfs_dictionary, label_vocab = pkl.load(reader)
    else:
        training_ds, lemma2synsets, mfs_dictionary, label_vocab = dataset_builder(encoder_name, training_paths,
                                                                                  label_mapper,
                                                                                  langs, mfs_file)
        training_ds.index_with(Vocabulary())
        training_iterator = get_bucket_iterator(training_ds, max_segments_in_batch)

        with open(os.path.join(".cache/", cached_dataset_file_name), "wb") as writer:
            pkl.dump((training_iterator, training_ds, lemma2synsets, mfs_dictionary, label_vocab), writer)
    return label_vocab, lemma2synsets, mfs_dictionary, training_ds, training_iterator


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
