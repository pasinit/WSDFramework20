from typing import List, Tuple

import numpy as np
from torch.nn import CrossEntropyLoss
from torch.nn.functional import cross_entropy
from torch.optim import Adam
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import os
from data.data_structures import Lemma2Synsets
from data.datasets import Vocabulary, WSDXmlInMemoryDataset, get_simplified_pos
from models.neural_wsd_models import WSDModel, TransformerFFWSDModel
import torch
import _pickle as pkl


class Trainer(object):
    def __init__(self, wsdmodel: WSDModel, optimiser: Optimizer):
        self.wsdmodel = wsdmodel
        self.wsdmodel.freeze_encoder_weights()
        self.wsdmodel.set_classifier_trainable()
        self.optimiser = optimiser
        self.criterion = CrossEntropyLoss()

    # def __accuracy(self, lemma_scores, labels):

    def train(self, training_data: WSDXmlInMemoryDataset, epochs,
              test_sets: List[Tuple[str, WSDXmlInMemoryDataset]] = None):
        # data_loader = DataLoader(training_data, batch_size, shuffle=True, collate_fn=WSDXmlInMemoryDataset.collate_fn)
        for epoch in range(epochs):
            bar = tqdm(training_data)
            losses = list()
            accs = list()
            for words, ids, lemmas, poss, labels in bar:

                self.optimiser.zero_grad()
                lemmapos = list()
                for bl, bp in zip(lemmas, poss):
                    blp = list()
                    for l, p in zip(bl, bp):
                        blp.append(l + "#" + get_simplified_pos(p))
                    lemmapos.append(blp)
                probabilities, lemma_bn_scores, mask = self.wsdmodel.forward(words, lemmapos)
                labels = [[x[0] if x is not None else -1 for x in y] for y in labels]

                gold_indices = [(torch.LongTensor(x) > 0).nonzero().flatten() for x in labels]
                new_probabilities = list()
                new_labels = list()
                new_mask = list()
                for indices, probs, m, lab in zip(gold_indices, probabilities, mask, labels):
                    new_mask.extend(torch.stack(m)[indices])
                    new_probabilities.extend(probs[indices].unbind())
                    new_labels.extend(torch.LongTensor(lab)[indices].tolist())
                new_mask = torch.stack(new_mask)
                new_probabilities = torch.stack(new_probabilities).to(self.wsdmodel.device) * new_mask
                new_labels = torch.LongTensor(new_labels).to(self.wsdmodel.device)
                loss = self.criterion(new_probabilities, new_labels)
                loss.backward()
                self.optimiser.step()
                # acc = self.__accuracy(lemma_scores, labels)
                losses.append(loss.item())
                # accs.append(acc)
                bar.set_postfix({"epoch": epoch, "loss": np.average(losses)})


if __name__ == "__main__":
    key_gold_path = "/home/tommaso/Documents/data/WSD_Evaluation_Framework/Training_Corpora/SemCor/semcor.gold.key.txt"
    xml_data_path = "/home/tommaso/Documents/data/WSD_Evaluation_Framework/Training_Corpora/SemCor/semcor.data.xml"
    key2bnid_path = "/home/tommaso/dev/PycharmProjects/DocumentWSD/resources/all_bn_wn_keys.txt"
    vocabulary = Vocabulary.vocabulary_from_gold_key_file(key_gold_path, key2bnid_path=key2bnid_path)
    lemma2synsetpath = "/media/tommaso/My Book/factories/output/lemma2bnsynsets.wn_part.en.txt"
    lemma2synsets = Lemma2Synsets(lemma2synsetpath)
    device = "cpu"
    model_name = "bert-base-cased"
    model = TransformerFFWSDModel(model_name, device, hidden_size=768, output_size=len(vocabulary),
                                  vocabulary=vocabulary, lemma2synsets=lemma2synsets)
    model.freeze_encoder_weights()
    adam = Adam(model.parameters())

    dataset_path = "resources/wsd_data/semcor.dataset.pkl"
    dataset_do_exists = os.path.exists(dataset_path)
    if not dataset_do_exists:
        dataset = WSDXmlInMemoryDataset(xml_data_path, key_gold_path, vocabulary,
                                        device=device,
                                        batch_size=32,
                                        key2bnid_path=key2bnid_path)

        with open(dataset_path, "wb") as writer:
            pkl.dump(dataset, writer)
    else:
        with open(dataset_path, "rb") as reader:
            dataset = pkl.load(reader)
    model.to(device)
    trainer = Trainer(model, adam)
    trainer.train(dataset, 10)
