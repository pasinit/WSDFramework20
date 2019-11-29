from allennlp.data import Vocabulary
from allennlp.data.iterators import BucketIterator
from tqdm import tqdm

from src.language_modelling_scorer.language_modelling import AutoHuggingfaceLM, AutoHuggingfaceIndexer
from src.language_modelling_scorer.lm_datasets import LMDataset
import torch
import allennlp.nn.util as nn_util
import jsonlines


class SentenceScorer(object):
    def __init__(self, model_name: str, max_seg_per_batch: int = 800):
        self.indexer = AutoHuggingfaceIndexer(model_name, use_starting_offsets=True)
        self.dataset = LMDataset(self.indexer)
        self.iterator = BucketIterator(
            sorting_keys=[("tokens", "num_tokens")],
            maximum_samples_per_batch=("tokens_length", max_seg_per_batch),
        )
        self.iterator.index_with(Vocabulary())
        self.model = AutoHuggingfaceLM(Vocabulary(), model_name).eval().to("cuda")

    def score_sentences(self, dataset_path: str, output_path: str):
        raw_data_generator = self.iterator(self.dataset.read(dataset_path),
                                           num_epochs=1,
                                           shuffle=False)
        counter = 0
        with jsonlines.open(output_path, mode="w") as writer:
            for batch in tqdm(raw_data_generator):
                input_dict = batch["tokens"]
                # if self.indexer.mask_token_id is not None:
                #     metadata = batch["metadata"]
                #     offsets = input_dict["tokens-offsets"]
                #     indices = [m["indices"] for m in metadata]
                #     indices = [[z for z, _ in elem] for elem in indices]
                #     indices_offsets = offsets[torch.arange(offsets.size(0)), indices]
                #     input_dict["tokens"][torch.arange(input_dict["tokens"].size(0)), indices_offsets] = self.indexer.mask_token_id
                batch = nn_util.move_to_device(batch, 0)
                out_dict = self.model(**batch)
                all_logits = out_dict["scores"]
                metadata_list = batch["metadata"]
                token_mapping = input_dict["tokens-offsets"]
                all_input_tokens = input_dict["tokens"]
                self.get_output(writer, all_input_tokens, all_logits, metadata_list, token_mapping,
                                          out_dict["loss"].view(batch["tokens"]["tokens"].size(0), -1))
                # for o in outputs:
                #     writer.write(o)
                counter += 1

                # if counter == 100:
                #     break

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
        token_logits = logits[mapping[words_indices]]
        word_seg_ids = token_ids[mapping[words_indices]]
        probs = torch.softmax(token_logits, -1)
        word_seg_scores = probs[torch.arange(probs.size(0)), word_seg_ids]
        # token_ids = input_ids[mapping]
        # token_preds = torch.softmax(token_logits, -1)
        # scores = token_preds[torch.arange(token_preds.size(0)), token_ids]
        words_seg_indices = mapping[words_indices]
        words_losses = losses[words_seg_indices]
        perplexity = torch.pow(2, -torch.mean(torch.log2(words_losses)))
        return list(zip(words, [x.item() for x in words_losses], [x.item() for x in word_seg_scores])), perplexity.item()


if __name__ == "__main__":
    scorer = SentenceScorer("gpt2")
    dataset_path = "data/wikipedia_sentences/wiki.en.cleanSplitSent.pasini.txt.idx.gz"
    output_path = "data/test.txt"
    scorer.score_sentences(dataset_path, output_path)
