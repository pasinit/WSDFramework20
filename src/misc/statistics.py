from collections import Counter

from src.data.dataset_utils import get_pos_from_key


def dataset_pos_stats(path):
    pos_counter = Counter()
    with open(path) as lines:
        for line in lines:
            fields = line.strip().split(" ")
            gold = fields[1]
            if gold.startswith("bn:"):
                pos = gold[-1]
                pos_counter[pos] += 1
            else:
                pos = get_pos_from_key(gold)
                pos_counter[pos] += 1
    tot = sum(pos_counter.values())
    to_print = [str(pos_counter[p]) for p in "n v a r".split()]
    print("\t" + str(tot) + "\t" + "\t".join(to_print))
import os
if __name__ == "__main__":
    root = "/home/tommaso/Documents/data/WSD_Evaluation_Framework_3.0/new_evaluation_datasets"
    print("Dataset\tALL\tNOUN\tVERB\tADJ\tADV")
    for test_name in "en en-no-sem10-no-sem07 en-coarse eu bg ca zh hr da nl et fr gl de hu it ja ko sl es".split():
        folder_name = "test-" + test_name
        gold_file = os.path.join(root, folder_name, folder_name + ".gold.key.txt")
        print(test_name, end="")
        dataset_pos_stats(gold_file)
