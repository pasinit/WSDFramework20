from pprint import pprint

from pytorch_lightning.loggers import WandbLogger

from src.evaluation import evaluate_answers

import pytorch_lightning as pl
import torch
from torch.utils.data.dataloader import DataLoader
from transformers import LxmertTokenizer

from src.data.multimodal_dataset import MultimodalTxtDataset
from src.models.multimodal_wsd_model import MultimodalWSDModel


class MultimodalWSDFinetuner(pl.LightningModule):
    def __init__(self, model, training_set, dev_set, training_params, dev_name):
        super().__init__()
        self.model = model
        self.hparams = training_params
        self.train_dataset = training_set
        self.dev_dataset = dev_set
        self.class2sense = self.train_dataset.class2sense
        self.dev_name = "/tmp/" + dev_name + ".predictions.txt"
        self.dev_path = dev_set.path_gold

    def forward(self, batch):
        return self.model(**batch)

    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        self.log("train_loss", outputs[0])
        return outputs[0]


    def validation_epoch_end(self, outputs):
        all_answers = dict()
        for _, b_logits, b_i_choices, b_choices, b_labels_mask, b_ids in outputs:
            # logits = b_logits.contiguous().view(-1, b_logits.shape[-1])
            # labels_mask = b_labels_mask.contiguous().view(-1).unsqueeze(-1)
            # logits = logits.masked_select(labels_mask).view(len(b_choices), -1)
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
        # print("OPENING FILE", self.dev_name)
        dev_golds = evaluate_answers.parse_file(self.dev_path)
        with open(self.dev_name, "w") as writer:
            for b_id, pred in all_answers.items():
                if len(b_id.split("_")) > 1:
                    b_id = b_id.split("_")[1]
                writer.write(b_id + " " + list(pred)[0] + "\n")

        # print(len(dev_golds), len(all_answers))
        if len(dev_golds) == len(all_answers):
            # print(all_answers)
            # print(dev_golds)
            accuracy = evaluate_answers.evaluate(all_answers, dev_golds, None, None, self.dev_name)
            self.log('val_accuracy', accuracy)

    def validation_step(self, batch, batch_idx):
        outputs = self(batch)
        self.log('val_loss', outputs[0])
        return (
        outputs[0], outputs[1], batch["choices"], batch["indexed_choices"], batch["labels_mask"], batch["instance_ids"])

    def get_optimizer_and_scheduler(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), self.hparams.lr)
        return optimizer

    def configure_optimizers(self):
        return self.get_optimizer_and_scheduler()

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size,
                          collate_fn=self.train_dataset.get_batch_fun(), num_workers=1)

    def val_dataloader(self):
        return DataLoader(self.dev_dataset, batch_size=self.hparams.batch_size,
                          collate_fn=self.dev_dataset.get_batch_fun(), num_workers=1)


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
    encoder_name = "unc-nlp/lxmert-base-uncased"
    sense_vocab_size = 206941
    finetune_encoder = True
    epochs = 50
    batch_accumulation = 16
    batch_size = 2
    lr = 2e-5
    params = TrainingParams()
    params["batch_size"] = batch_size
    params["lr"] = lr
    params["batch_accumulation"] = batch_accumulation
    encoder_tokenizer = LxmertTokenizer.from_pretrained(encoder_name)

    semcor_path = "/home/tommaso/Documents/data/WSD_Evaluation_Framework/Training_Corpora/SemCor/semcor.data.xml"
    dev_path = "/home/tommaso/Documents/data/WSD_Evaluation_Framework/Evaluation_Datasets/semeval2007/semeval2007.data.xml"
    wordnet_sense_index_path = "/opt/WordNet-3.0/dict/index.sense"
    train_tokid2imgid_path = "data2/training_data/multimodal/tok2image.semcor.restricted.txt"
    train_imgfeat_path = "data2/training_data/multimodal/img.gcc.semcor.new.npz"
    dev_tokid2imgid_path = "data2/training_data/multimodal/tok2image.semeval2007.restricted.txt"
    dev_imgfeat_path = "data2/training_data/multimodal/img.gcc.semeval2007.other.npz"

    dataset = MultimodalTxtDataset(encoder_tokenizer,
                                   semcor_path, train_tokid2imgid_path, train_imgfeat_path, wordnet_sense_index_path)

    dev_dataset = MultimodalTxtDataset(encoder_tokenizer,
                                       dev_path, dev_tokid2imgid_path, dev_imgfeat_path, wordnet_sense_index_path)
    model = MultimodalWSDModel(sense_vocab_size, encoder_name, finetune_encoder=finetune_encoder)
    wandb_logger = WandbLogger("lxmert_wsd",
                               project="multimodal_wsd",
                               offline=False,
                               log_model=True,
                               save_dir="data4/")
    trainer = pl.Trainer(gpus=1, precision=32, max_epochs=epochs,
                         accumulate_grad_batches=batch_accumulation,
                         logger=[wandb_logger])  # , limit_val_batches=0.1)

    finetuner = MultimodalWSDFinetuner(model, dataset, dev_dataset, params, "semeval2007")
    trainer.fit(finetuner)