from allennlp.data import Vocabulary
from allennlp.data.iterators import BucketIterator
from numpy import random
from tqdm import tqdm

from src.language_modelling_scorer.language_modelling import AutoHuggingfaceLM, AutoHuggingfaceIndexer
from src.language_modelling_scorer.lm_datasets import LMDataset
import torch
import allennlp.nn.util as nn_util
import jsonlines


class SentenceScorer(object):
    def __init__(self, model_name: str, max_segments_per_batch: int = 800, mask_rate: float = 0.3, **kwargs):
        self.indexer = AutoHuggingfaceIndexer(model_name, use_starting_offsets=True, **kwargs)
        self.dataset = LMDataset(self.indexer, **kwargs)
        self.iterator = BucketIterator(
            sorting_keys=[("tokens", "num_tokens")],
            maximum_samples_per_batch=("tokens_length", max_segments_per_batch),
        )
        self.model_name = model_name
        self.iterator.index_with(Vocabulary())
        self.mask_rate = mask_rate
        self.model = AutoHuggingfaceLM(Vocabulary(), model_name).eval().to("cuda")
        self.is_masked_lm = "gpt" not in self.model_name

    def prepare_masked_input(self, batch):
        for tokens in batch["tokens"]["tokens"]:
            t_l = len(tokens)
            i = t_l - 1
            while i >= 0:
                if tokens[i] != self.indexer.tokeniser.pad_token_id:
                    break
                i -= 1

            integers_to_mask = random.choice(list(range(0, i + 1)), int(self.mask_rate * i), replace=False)
            tokens[integers_to_mask] = self.indexer.mask_token_id

    def score_sentences(self, dataset_path: str, output_path: str):
        raw_data_generator = self.iterator(self.dataset.read(dataset_path),
                                           num_epochs=1,
                                           shuffle=False)
        counter = 0
        with jsonlines.open(output_path, mode="w") as writer:
            for batch in tqdm(raw_data_generator):
                input_dict = batch["tokens"]
                if not "gpt2" in self.model_name:
                    self.prepare_masked_input(batch)
                batch = nn_util.move_to_device(batch, 0)
                with torch.no_grad():
                    out_dict = self.model(**batch)
                all_logits = out_dict["scores"]
                metadata_list = batch["metadata"]
                token_mapping = input_dict["tokens-offsets"]
                all_input_tokens = input_dict["tokens"]
                self.get_output(writer, all_input_tokens, all_logits, metadata_list, token_mapping,
                                out_dict["loss"].view(batch["tokens"]["tokens"].size(0), -1))
                counter += 1

    def get_output(self, writer, all_input_tokens, all_logits, metadata_list, token_mapping, all_losses):
        outputs = list()
        for mapping, metadata, input_tokens, losses, logits in zip(token_mapping,
                                                                   metadata_list,
                                                                   all_input_tokens, all_losses, all_logits):
            sid = metadata["sid"]
            indices = metadata["indices"]
            words = metadata["words"]
            word_scores, perplexity = self.get_scores(input_tokens, logits, losses, mapping, words,
                                                      [x for x, _ in indices])
            writer.write({"sentence_id": sid, "perplexity": perplexity,
                          "words": [
                              {"word": word.replace(" ", "_"), "loss": score, "probability": prob}
                              for word, score, prob in
                              word_scores]
                          })
        return outputs

    def get_scores(self, token_ids, logits, losses, mapping, words, words_indices):
        seg_indices = mapping[words_indices]
        if not self.is_masked_lm:
            ## then I need to take the predictions from the previou token
            seg_indices = seg_indices - 1
        token_logits = logits[seg_indices]
        if not self.is_masked_lm:
            ## then I need take the segment id back to the original one to extract the right
            ## probability from the distribution
            seg_indices = seg_indices + 1
        word_seg_ids = token_ids[seg_indices]
        probs = torch.softmax(token_logits, -1)
        word_seg_scores = probs[torch.arange(probs.size(0)), word_seg_ids]
        words_losses = losses[seg_indices]
        perplexity = torch.pow(2, -torch.mean(torch.log2(words_losses)))
        return list(
            zip(words, [x.item() for x in words_losses], [x.item() for x in word_seg_scores])), perplexity.item()


if __name__ == "__main__":
    # scorer = SentenceScorer("xlm-mlm-100-1280", one_elem_per_word=True)
    scorer = SentenceScorer("gpt2")
    dataset_path = "data/wikipedia_sentences/wiki.en.cleanSplitSent.pasini.txt.idx.gz"
    output_path = "data/test.txt"
    scorer.score_sentences(dataset_path, output_path)
