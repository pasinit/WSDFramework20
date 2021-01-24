import logging
from pprint import pprint, pformat
import os
from src.models.generic_transformer_wsd_model import TransformerWSDModel

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import torch
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data.dataloader import DataLoader
from transformers import AutoTokenizer
from transformers import LxmertTokenizerFast
from transformers.modeling_auto import AutoModel
from transformers.tokenization_bert import BertTokenizerFast

from src.data.multimodal_dataset import MultimodalTxtDataset
from src.evaluation import evaluate_answers
from src.models.multimodal_wsd_model import BertCrossAttentionWSDModel, BertFusionWSDModel, MultimodalWSDModel
from torch.optim.lr_scheduler import *

LEVEL = logging.DEBUG
logger = logging.getLogger(__name__)
logger.setLevel(LEVEL)
ch = logging.StreamHandler()
ch.setLevel(LEVEL)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


class MultimodalWSDFinetuner(pl.LightningModule):
    def __init__(self, model, training_set, dev_set,
                 test_datset, training_params, dev_name, test_name, steps_without_images=-1):
        super().__init__()
        self.model = model
        self.hparams = training_params
        self.train_dataset = training_set
        self.dev_dataset = dev_set
        self.test_dataset = test_dataset
        self.class2sense = self.train_dataset.class2sense
        self.dev_name = "/tmp/" + dev_name + ".predictions.txt"
        self.test_name = "/tmp/" + test_name + ".predictions.txt"
        self.dev_path = dev_set.path_gold
        self.test_path = test_dataset.path_gold
        self.steps_without_images = steps_without_images

    def forward(self, batch):
        return self.model(**batch)

    def training_step(self, batch, batch_idx):
        # if self.global_step < self.steps_without_images:
        #     batch["visual_pos"] = torch.zeros_like(batch["visual_pos"])
        #     batch["visual_feats"] = torch.zeros_like(batch["visual_feats"])
        #     batch["visual_attention_mask"] = torch.zeros_like(
        #         batch["visual_attention_mask"])
        #     batch["visual_attention_mask"][:, 0] = 1
        # elif self.global_step == self.steps_without_images:
        #     logger.info("images are now provided")
        outputs = self(batch)
        self.log("train_loss", outputs[0])
        self.log(
            "lr", self.trainer.lr_schedulers[0]["scheduler"]._last_lr[0], prog_bar=True, on_step=True)
        return outputs[0]

    def evaluate(self, outputs, dataset_name, dataset_path, is_validation=True):
        all_answers = dict()
        for _, b_logits, b_i_choices, b_choices, b_labels_mask, b_ids in outputs:
            b_choices = [list(x) for x in b_choices]
            choice_mask = torch.zeros_like(b_logits) - 1e9
            for i, x in enumerate(b_choices):
                choice_mask[i][x] = 0
            b_logits += choice_mask
            predictions = torch.argmax(b_logits, -1)
            predictions = [{self.class2sense[x.item()]} for x in predictions]
            b_ids = [x.split("_")[-1] for x in b_ids]
            dev_answers = dict(zip(b_ids, predictions))
            all_answers.update(dev_answers)
        dev_golds = evaluate_answers.parse_file(dataset_path)
        with open(dataset_name, "w") as writer:
            for b_id, pred in all_answers.items():
                if len(b_id.split("_")) > 1:
                    b_id = b_id.split("_")[1]
                writer.write(b_id + " " + list(pred)[0] + "\n")
        if len(dev_golds) == len(all_answers):
            accuracy = evaluate_answers.evaluate(
                all_answers, dev_golds, None, None)
            if is_validation:
                self.log('val_accuracy', accuracy["ALL"])
            else:
                self.log('test_accuracy', accuracy["ALL"])

    def validation_epoch_end(self, outputs):
        self.evaluate(outputs, self.dev_name, self.dev_path)

    def test_epoch_end(self, outputs):
        self.evaluate(outputs, self.test_name,
                      self.test_path, is_validation=False)

    def validation_step(self, batch, batch_idx):
        outputs = self(batch)
        self.log('val_loss', outputs[0])
        return (
            outputs[0], outputs[1], batch["choices"], batch["indexed_choices"], batch["labels_mask"],
            batch["instance_ids"])

    def test_step(self, batch, **kwargs):
        outputs = self(batch)
        self.log('test_loss', outputs[0])
        return (
            outputs[0], outputs[1], batch["choices"], batch["indexed_choices"], batch["labels_mask"],
            batch["instance_ids"])

    def get_warmup_lr_fn(self):
        rescaled_max_steps = self.hparams.warmup_steps // self.hparams.batch_accumulation
        def fun(steps):
            lr_scale = 1.0
            if steps < rescaled_max_steps:
                lr_scale = min(1., float(steps) / rescaled_max_steps)
            return lr_scale
        return fun

    def get_optimizer_and_scheduler(self):
        params = self.model.named_parameters()
        params = [
            {"params": [p for k, p in params if not k.startswith("encoder.")],
             "weight_decay": 0.01},
            {"params": [p for k, p in params if k.startswith("encoder.")],
             "weight_decay": 0.0}
        ]

        optimizer = torch.optim.AdamW(params, self.hparams.lr)
        fn = self.get_warmup_lr_fn()
        scheduler = LambdaLR(optimizer, fn)
        scheduler = {
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': 1
        }

        return [optimizer], [scheduler]

    def configure_optimizers(self):
        return self.get_optimizer_and_scheduler()

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size,
                          collate_fn=self.train_dataset.get_batch_fun(), num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.dev_dataset, batch_size=self.hparams.batch_size,
                          collate_fn=self.dev_dataset.get_batch_fun(), num_workers=8)


class TrainingParams(dict):
    def __init__(self, **kwargs):
        super().__init__()
        self.update(kwargs)
        # self.params = kwargs

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(TrainingParams, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(TrainingParams, self).__delitem__(key)
        del self.__dict__[key]


if __name__ == "__main__":
    torch.random.manual_seed(42)
    np.random.seed(42)

    encoder_name = "bert-base-cased"
    model_name = encoder_name.split("/")[-1] + "-fusion-x-attention"
    sense_vocab_size = 206941
    finetune_encoder = False
    epochs = 100
    batch_accumulation = 8
    batch_size = 8
    lr = 0.00002
    params = TrainingParams()
    params["batch_size"] = batch_size
    params["lr"] = lr
    params["batch_accumulation"] = batch_accumulation
    params["steps_without_images"] = 0
    params["img_retriver"] = "sbert_finetuned_bn_glosses"  # "sbert_baseline"
    params["img_corpus"] = "gcc"
    params["warmup_steps"] = 4553 * 50
    encoder_tokenizer = AutoTokenizer.from_pretrained(encoder_name, fast=True)

    semcor_path = "/home/tommaso/Documents/data/WSD_Evaluation_Framework/Training_Corpora/SemCor/semcor.data.xml"
    dev_path = "/home/tommaso/Documents/data/WSD_Evaluation_Framework/Evaluation_Datasets/semeval2007/semeval2007.data.xml"
    test_path = "/home/tommaso/Documents/data/WSD_Evaluation_Framework/Evaluation_Datasets/ALL/ALL.data.xml"
    wordnet_sense_index_path = "/opt/WordNet-3.0/dict/index.sense"

    train_tokid2imgid_path = "/home/tommaso/dev/PycharmProjects/multimodal_wsd_bianca/data/in/new_data_iacer/txt_img_mapping/{}/semcor_{}.txt".format(
        params["img_retriver"], params["img_corpus"])
    dev_tokid2imgid_path = "/home/tommaso/dev/PycharmProjects/multimodal_wsd_bianca/data/in/new_data_iacer/txt_img_mapping/{}/semeval2007_{}.txt".format(
        params["img_retriver"], params["img_corpus"])

    imgfeat_path = f"/home/tommaso/dev/PycharmProjects/multimodal_wsd_bianca/data/in/new_data_iacer/frcnn_features/{params['img_corpus']}/top1_images.wsd_gcc_images.npz"

    img_features_files = np.load(imgfeat_path)

    dataset = MultimodalTxtDataset(encoder_tokenizer,
                                   semcor_path,
                                   train_tokid2imgid_path, img_features_files,
                                   wordnet_sense_index_path)
    dev_dataset = MultimodalTxtDataset(encoder_tokenizer,
                                       dev_path, dev_tokid2imgid_path, img_features_files,
                                       wordnet_sense_index_path)
    test_dataset = MultimodalTxtDataset(encoder_tokenizer,
                                        test_path,
                                        dev_tokid2imgid_path, img_features_files,
                                        wordnet_sense_index_path)
    # model = MultimodalWSDModel(
    # sense_vocab_size, encoder_name, finetune_encoder=finetune_encoder, cache=True)
    # model = BertFusionWSDModel(sense_vocab_size, encoder_name,
    #                            img_feature_size=2048,
    #                            finetune_encoder=finetune_encoder,
    #                            cache=True)
    model = BertCrossAttentionWSDModel(sense_vocab_size, encoder_name,
                                       img_feature_size=2048,
                                       finetune_encoder=finetune_encoder,
                                       cache=True)
    wandb_logger = WandbLogger(model_name + "_" + params["img_retriver"] + "_" + params["img_corpus"] + "_wsd",
                               project="multimodal_wsd",
                               offline=False,
                               log_model=True,
                               save_dir="data4/")
    checkpoint_dir = os.path.join(wandb_logger.experiment.dir, "checkpoints/")
    os.makedirs(checkpoint_dir)
    checkpointer = ModelCheckpoint(
        os.path.join(checkpoint_dir, "{global_step}"), monitor="val_accuracy",
        save_top_k=1
    )
    trainer = pl.Trainer(gpus=1, precision=16, max_epochs=epochs,
                         accumulate_grad_batches=batch_accumulation,
                         logger=[wandb_logger],
                         num_sanity_val_steps=10,
                         checkpoint_callback=checkpointer
                         )  # , limit_val_batches=0.1)

    finetuner = MultimodalWSDFinetuner(model, dataset, dev_dataset,
                                       test_dataset,
                                       params,
                                       "semeval2007",
                                       "ALL",
                                       steps_without_images=params.steps_without_images)

    logger.info(pformat(params))
    trainer.fit(finetuner)
    trainer.test()
