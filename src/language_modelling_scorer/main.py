from argparse import ArgumentParser

from src.language_modelling_scorer.scorers import SentenceScorer
import pprint


def main(args):
    scorer = SentenceScorer(**vars(args))
    scorer.score_sentences(args.sentence_file, args.output_file)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--sentence_file", required=True)
    parser.add_argument("--model_name", required=True, choices=["gpt2", "xlm-mlm-100-1280"])
    parser.add_argument("--one_elem_per_word", required=True, type=bool, help="define whether the dataset reader"
                                                                             "has to return an instance for each"
                                                                             "target word or for each sentence. Typically,"
                                                                             "xlm needs one instance per word, so set it to True"
                                                                             "if model_name==xlm-*")
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--max_segments_per_batch", default=800, type=int,
                        help="number of segments to include in a single batch.")
    parser.add_argument("--min_tokens", default=5, type=int)
    parser.add_argument("--max_tokens", default=70, type=int)
    parser.add_argument("--lazy", default=True, type=bool)
    parser.add_argument("--mask_rate", default=0.3, type=float,
                        help="probability for a wordpiece in a sentence to be masked"
                             "when computing the perplexity for a masked-lm")

    args = parser.parse_args()
    pprint.pprint(args)
    main(args)
