from argparse import ArgumentParser

from src.language_modelling_scorer.scorers import SentenceScorer


def main(args):
    scorer = SentenceScorer(args.model_name, args.max_segments_per_batch)
    scorer.score_sentences(args.sentence_file, args.output_file)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--sentence_file", required=True)
    parser.add_argument("--model_name", required=True, choices=["gpt2"])
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--max_segments_per_batch", default=800, type=int,
                        description="number of segments to include in a single batch.")

    args = parser.parse_args()
    main(args)
