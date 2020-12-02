DATASET_NAMES = [
    "senseval2",
    "senseval3",
    "semeval2007",
    "semeval2007_coarse",
    "semeval2010",
    "semeval2013",
    "semeval2015",
    "ALL",
    "ALL_no_semeval2007",
    "semeval2013_it",
    "semeval2013_it_wnfilter",
    "semeval2015_it",
    "semeval2015_it_wnfilter",
    "wiki_dev_it",
    "wiki_dev_it_wnfilter",
    "wiki_test_it",
    "wiki_test_it_wnfilter",

    "semeval2013_es",
    "semeval2013_es_wnfilter",
    "semeval2015_es",
    "semeval2015_es_wnfilter",
    "wiki_dev_es",
    "wiki_dev_es_wnfilter",
    "wiki_test_es",
    "wiki_test_es_wnfilter",
    "semeval2013_fr",
    "semeval2013_fr_wnfilter",
    "wiki_dev_fr",
    "wiki_dev_fr_wnfilter",
    "wiki_test_fr",
    "wiki_test_fr_wnfilter",

    "semeval2013_de",
    "semeval2013_de_wnfilter",
    "wiki_dev_de",
    "wiki_dev_de_wnfilter",
    "wiki_test_de",
    "wiki_test_de_wnfilter",
]
import os


def stats(directory):
    for dataset in DATASET_NAMES:
        path = os.path.join(directory, dataset, dataset + ".gold.key.txt")
        print("{}\t{}".format(dataset, len(open(path).readlines())))


def get_wn_ids(golpath):
    with open(golpath) as lines:
        count = 0
        for line in lines:
            fields = [x for x in line.strip().split(" ")[2:] if x.startswith("bn:")]
            if any(int(x[3:-1]) < 117660 or x == "bn:14866890n" for x in fields):
                count += 1
        print(golpath.split("/")[-1], count)


if __name__ == "__main__":
    # stats("/home/tommaso/Documents/data/WSD_Evaluation_Framework_2.0/Evaluation_Datasets")
    get_wn_ids("/home/tommaso/Documents/data/SemEval/SemEval2015/keys/gold_keys/ES/semeval-2015-task-13-es-WSD.key")
