from argparse import ArgumentParser

from src.language_modelling_scorer.scorers import SentenceScorer


def main(args):
    scorer = SentenceScorer(args.model_name)
    scorer.score_sentences(args.sentence_file, args.output_file)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--sentence_file", required=True)
    parser.add_argument("--model_name", required=True, choices=["gpt2", "xlm"])
    parser.add_argument("--ouput_file", required=True)
    args = parser.parse_args()
    main(args)
