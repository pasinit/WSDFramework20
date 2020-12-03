import logging
from pprint import pprint, pformat
from src.models.generic_transformer_wsd_model import TransformerWSDModel

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data.dataloader import DataLoader
from transformers import LxmertTokenizerFast
from transformers.modeling_auto import AutoModel
from transformers.tokenization_bert import BertTokenizerFast

from src.data.multimodal_dataset import MultimodalTxtDataset
from src.evaluation import evaluate_answers
from src.models.multimodal_wsd_model import MultimodalWSDModel

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
    def __init__(self, model, training_set, dev_set, training_params, dev_name, steps_without_images=-1):
        super().__init__()
        self.model = model
        self.hparams = training_params
        self.train_dataset = training_set
        self.dev_dataset = dev_set
        self.class2sense = self.train_dataset.class2sense
        self.dev_name = "/tmp/" + dev_name + ".predictions.txt"
        self.dev_path = dev_set.path_gold
        self.steps_without_images = steps_without_images

    def forward(self, batch):
        return self.model(**batch)

    def training_step(self, batch, batch_idx):
        if self.global_step < self.steps_without_images:
            batch["visual_pos"] = torch.zeros_like(batch["visual_pos"])
            batch["visual_feats"] = torch.zeros_like(batch["visual_feats"])
            batch["visual_attention_mask"] = torch.zeros_like(
                batch["visual_attention_mask"])
            batch["visual_attention_mask"][:, 0] = 1
        outputs = self(batch)
        self.log("train_loss", outputs[0])
        return outputs[0]

    def validation_epoch_end(self, outputs):
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
        dev_golds = evaluate_answers.parse_file(self.dev_path, self.dev_dataset.sense2offset)
        with open(self.dev_name, "w") as writer:
            for b_id, pred in all_answers.items():
                if len(b_id.split("_")) > 1:
                    b_id = b_id.split("_")[1]
                writer.write(b_id + " " + list(pred)[0] + "\n")

        # print(len(dev_golds), len(all_answers))
        if len(dev_golds) == len(all_answers):
            # print(all_answers)
            # print(dev_golds)
            accuracy = evaluate_answers.evaluate(
                all_answers, dev_golds, None, None, self.dev_name)
            self.log('val_accuracy', accuracy, prog_bar=True)
            

    def validation_step(self, batch, batch_idx):
        outputs = self(batch)
        self.log('val_loss', outputs[0])
        return (
            outputs[0], outputs[1], batch["choices"], batch["indexed_choices"], batch["labels_mask"],
            batch["instance_ids"])

    def get_optimizer_and_scheduler(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), self.hparams.lr)
        return optimizer

    def configure_optimizers(self):
        return self.get_optimizer_and_scheduler()

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size,
                          collate_fn=self.train_dataset.get_batch_fun(), num_workers=0, 
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.dev_dataset, batch_size=self.hparams.batch_size,
                          collate_fn=self.dev_dataset.get_batch_fun(), 
                          num_workers=0)


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
    encoder_name = "bert-base-cased"#"unc-nlp/lxmert-base-uncased"
    # sense_vocab_size = 206941
    finetune_encoder = False
    epochs = 50
    batch_accumulation = 16
    batch_size = 4
    lr = 2e-5
    params = TrainingParams()
    params["batch_size"] = batch_size
    params["lr"] = lr
    params["batch_accumulation"] = batch_accumulation
    params["steps_without_images"] = 0
    encoder_tokenizer = BertTokenizerFast.from_pretrained(encoder_name)
    semcor_path = "data/training_data/en_training_data/semcor/semcor.data.xml"
    dev_path = "resources/evaluation_framework_3.0/new_evaluation_datasets/dev-en/dev-en.data.xml"
    wordnet_sense_index_path = "/opt/WordNet-3.0/dict/index.sense"

    train_tokid2imgid_path = "data/in/wsd/semcor_sentence_image_map.txt"
    dev_tokid2imgid_path = "data/in/wsd/semeval2007_sentence_image_map.txt"

    imgfeat_path = "data/in/wsd/wsd_gcc_images.npz"

    img_features_files = np.load(imgfeat_path)

    dataset = MultimodalTxtDataset(encoder_tokenizer,
                                   semcor_path, train_tokid2imgid_path, img_features_files,
                                   wordnet_sense_index_path)
    dev_dataset = MultimodalTxtDataset(encoder_tokenizer,
                                       dev_path, dev_tokid2imgid_path, img_features_files,
                                       wordnet_sense_index_path,
                                       inventory=dataset.inventory,
                                       class2sense = dataset.class2sense,
                                       sense2class = dataset.sense2class,
                                       sense2offset = dataset.sense2offset)

    model = TransformerWSDModel(encoder_name, dataset.inventory_size, finetune_encoder)
    wandb_logger = WandbLogger(encoder_name+"_wsd",
                               project="multimodal_wsd",
                               offline=False,
                               log_model=True,
                               save_dir="data/",
                               entity="research")
    trainer = pl.Trainer(gpus=1, precision=16, max_epochs=epochs,
                         accumulate_grad_batches=batch_accumulation,
                         logger=[wandb_logger],
                         num_sanity_val_steps=0
                         )  # , limit_val_batches=0.1)

    finetuner = MultimodalWSDFinetuner(model, dataset, dev_dataset, params, "semeval2007",
                                       steps_without_images=params.steps_without_images)

    logger.info(pformat(params))
    trainer.fit(finetuner)
