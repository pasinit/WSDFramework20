import _pickle as pkl
import os
import socket
from argparse import ArgumentParser

import numpy as np
import torch
import wandb
import yaml
from allennlp.data import Vocabulary, DataLoader, allennlp_collate
from allennlp.data.samplers import BucketBatchSampler
from allennlp.training import GradientDescentTrainer
from allennlp.training.optimizers import AdamOptimizer
from allennlp_training_callbacks.callbacks import TestAndWrite

from src.data.dataset_utils import get_dataset_with_labels_from_data, get_wnoffsets_dataset, get_sensekey_dataset, \
    get_bnoffsets_dataset, get_label_mapper
from src.evaluation.evaluate_model import evaluate_datasets
from src.misc.wsdlogging import get_info_logger
from src.models.neural_wsd_models import AllenWSDModel, WSDOutputWriter

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
    finetune_embedder = model_config.get("finetune", False)
    learning_rate = float(model_config["learning_rate"])
    cache_instances = training_config["cache_instances"]
    num_epochs = training_config["num_epochs"]
    wandb.init(config=config, project="wsd_framework", tags=[socket.gethostname(), model_name, ",".join(langs)])
    if dev_name is None:
        logger.warning("No dev name set... In this way I won't save in best.th the best model according to the "
                       "development set. best.th will contain the weights of the model at its last epoch")
    device_int = 0 if device == "cuda" else -1
    test_paths = [os.path.join(test_data_root, name, name + ".data.xml") for name in test_names]
    training_paths = train_data_root
    outpath = os.path.join(outpath, model_name)
    build_outpath_subdirs(outpath)

    if label_from_training:
        logger.warn("using labels from training is highly discouraged as the method to generate the dataset is not "
                    "longer maintained.")
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
    all_labels = list()
    for f in training_paths:
        with open(f.replace(".data.xml", ".gold.key.txt")) as reader:
            all_labels.extend([l.split(" ")[1] for l in reader])
    label_mapper = get_label_mapper(target_inventory=sense_inventory, labels=all_labels)

    training_ds, lemma2synsets, mfs_dictionary, label_vocab = dataset_builder(model_name, training_paths, label_mapper,
                                                                              langs, mfs_file)
    dev_ds = None
    if dev_name is not None:
        dev_path = test_paths[test_names.index(dev_name)]
        dev_ds, *_ = dataset_builder(model_name, dev_path, label_mapper, langs, mfs_file)
        dev_ds.index_with(Vocabulary())
    test_dss = [dataset_builder(model_name, t, label_mapper, langs, mfs_file)[0] for t in test_paths]
    training_ds.index_with(Vocabulary())
    for td in test_dss:
        td.index_with(Vocabulary())

    model = AllenWSDModel.get_transformer_based_wsd_model(model_name, len(label_vocab), lemma2synsets, device_int,
                                                          label_vocab, training_ds.pad_token_id,
                                                          vocab=Vocabulary(), mfs_dictionary=mfs_dictionary,
                                                          cache_vectors=cache_instances,
                                                          finetune_embedder=finetune_embedder,
                                                          model_path=model_config.get("model_path", None))
    #####################################################
    # NEDED so to not split sentences in the test data. #
    reader.max_sentence_len = 512
    reader.sliding_window_size = 512
    #####################################################
    # logger.info("loading test data...")
    # tests_dss = [reader.read(test_path, label_mapper_getter=get_label_mapper) for test_path in test_paths]
    # iterator = BasicIterator(maximum_samples_per_batch=("tokens_length", max_segments_in_batch),
    #                          cache_instances=True
    #                          )
    # iterator = BucketIterator(
    #     biggest_batch_first=True,
    #     sorting_keys=[("tokens", "num_tokens")],
    #     maximum_samples_per_batch=("tokens_length", max_segments_in_batch),
    #     cache_instances=True,
    #     # instances_per_epoch=10
    # )
    # valid_iterator = BucketIterator(
    #     maximum_samples_per_batch=("tokens_length", max_segments_in_batch),
    #     biggest_batch_first=True,
    #     sorting_keys=[("tokens", "num_tokens")],
    #     cache_instances=True
    #     # instances_per_epoch=10
    #
    # )
    training_iterator = DataLoader(training_ds, batch_sampler=BucketBatchSampler(training_ds, 32, ["tokens"]),
                                   collate_fn=allennlp_collate, batches_per_epoch=1)
    # test_iterators = [DataLoader(td, batch_sampler=BucketBatchSampler(td, 32, ["tokens"]),
    #                              collate_fn=lambda x: Batch(x)) for td in test_dss]
    dev_iterator = None
    if dev_ds is not None:
        dev_iterator = DataLoader(dev_ds, batch_sampler=BucketBatchSampler(dev_ds, 32, ["tokens"]),
                                  collate_fn=allennlp_collate)
    writers = [WSDOutputWriter(os.path.join(outpath, "predictions", name + ".predictions.txt"), label_vocab.itos)
               for name in test_names]
    callbacks = [TestAndWrite(test_iterator=DataLoader(td, batch_sampler=BucketBatchSampler(td, 32, ["tokens"]),
                                                       collate_fn=allennlp_collate),
                              output_writer=writer,
                              name=name,
                              wandb=True,
                              is_dev=name == dev_name if dev_name is not None else False)
                 for name, td, writer in zip(test_names, test_dss, writers)]

    trainer = GradientDescentTrainer(model=model,
                                     optimizer=AdamOptimizer(model.named_parameters(), lr=learning_rate),
                                     data_loader=training_iterator,
                                     cuda_device=device_int,
                                     num_epochs=num_epochs,
                                     validation_data_loader=dev_iterator,
                                     validation_metric="+f1_mfs" if mfs_file else "+f1",
                                     epoch_callbacks=callbacks,
                                     serialization_dir=os.path.join(outpath, "checkpoints"),
                                     # checkpointer=Checkpointer(os.path.join(outpath, "checkpoints"), num_serialized_models_to_keep=100),
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
                      mfs_dictionary, mfs_dictionary is not None, os.path.join(outpath, "evaluation"),
                      training_ds.tokenizer.tokenizer.pad_token_id,
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
