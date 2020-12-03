from argparse import ArgumentParser
from collections import Counter

from src.data.dataset_utils import get_pos_from_key
import os


def parse_file(path, mapping=None):
    id2ans = dict()
    with open(path) as lines:
        for line in lines:
            fields = line.strip().split(" ")
            id, *answers = fields
            if mapping is not None:
                answers = [mapping[x] for x in answers]
            id2ans[id] = set(answers)
    return id2ans


def get_pos(label):
    if label.startswith("bn:") or label.startswith("wn:"):
        return label[-1]
    return get_pos_from_key(label)


def get_bn_labels(labels, wnkey2bn):
    bn_labels = set()
    for l in labels:
        if l.startswith("bn:"):
            bn_labels.add(l)
            continue
        else:
            bn_labels.add(wnkey2bn[l])
    return bn_labels


def evaluate(answers, golds, by_pos, wnkey2bn, dataset):
    correct = 0
    tot = 0
    correct_by_pos = Counter()
    tot_by_pos = Counter()
    for id in golds.keys():
        ans = answers[id]
        labels = golds[id]
        if wnkey2bn is not None:
            labels = get_bn_labels(labels, wnkey2bn)
        pos = get_pos(list(labels)[0])

        if len(ans & labels) > 0:
            correct += 1
            correct_by_pos[pos] += 1

        tot += 1
        tot_by_pos[pos] += 1
    accuracy = correct / tot
    # print("F1:", accuracy)
    results = []
    if by_pos:
        for p in "n,v,a,r".split(","):
            p_tot = tot_by_pos[p]
            p_cor = correct_by_pos[p]
            if p_tot > 0:
                results.append(p_cor / p_tot)
            else:
                results.append(-1.0)
            # print("{} F1: {}".format(p.upper(), p_cor / p_tot))
    #print(dataset + "\t" + "\t".join(["{:.4}".format(x * 100) for x in [accuracy] + results]))
    return accuracy

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--answer_dir", required=True)
    parser.add_argument("--gold_dir", required=True)
    parser.add_argument("--by_pos", action="store_true", default=False)

    args = parser.parse_args()
    all_bn_wn_keys_file = "/home/tommaso/dev/PycharmProjects/WSDframework/resources/mappings/all_bn_wn_keys.txt"
    wnkey2bn = dict()
    with open(all_bn_wn_keys_file) as lines:
        for line in lines:
            fields = line.strip().split("\t")
            bnid = fields[0]
            wnkeys = fields[1:]
            for wnkey in wnkeys:
                wnkey2bn[wnkey] = bnid
    print("all\tn\tv\ta\tr".upper())

    for dataset in "test-en,test-en-coarse,test-en-no-sem10-no-sem07,dev-en,test-it,dev-it,test-es,dev-es,test-fr,dev-fr,test-de,dev-de,test-zh,dev-zh,test-gl,dev-gl,test-hr,\
dev-hr,test-da,dev-da,test-et,dev-et,test-ja,dev-ja,test-hu,dev-hu,test-bg,dev-bg,test-eu,dev-eu,test-ca,dev-ca,test-ko,\
dev-ko,test-sl,dev-sl,test-nl,dev-nl".split(","):
        answer_file = os.path.join(args.answer_dir, dataset + ".predictions.txt")
        gold_file = os.path.join(args.gold_dir, dataset, dataset + ".gold.key.txt")
        by_pos = args.by_pos

        id2answer = parse_file(answer_file)
        id2gold = parse_file(gold_file)
        evaluate(id2answer, id2gold, by_pos, wnkey2bn, dataset)
