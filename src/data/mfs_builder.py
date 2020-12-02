from argparse import ArgumentParser

from src.data.dataset_utils import get_pos_from_key

WN_INDEX_SENSE_PATH = "/opt/WordNet-3.0/dict/index.sense"


def build_wn_mfs(out_path, gold_key_field):
    with open(WN_INDEX_SENSE_PATH) as lines, open(out_path, "w") as writer:
        for line in lines:
            fields = line.strip().split(" ")

            if int(fields[2]) == 1:
                sensekey = fields[0]
                lemma = sensekey.split("%")[0]
                pos = get_pos_from_key(sensekey)
                lemmapos = lemma + "#" + pos
                gold = fields[gold_key_field]
                writer.write("{}\t{}\n".format(lemmapos, gold))


def build_wn_sensekey_mfs(out_path):
    build_wn_mfs(out_path, 0)


def build_wn_offset_mfs(out_path):
    build_wn_mfs(out_path, 1)


def build_wn_bnids_mfs(out_path):
    wnkey2bnid = dict()
    with open("resources/mappings/all_bn_wn_keys.txt") as lines:
        for line in lines:
            fields = line.strip().split("\t")
            bnid = fields[0]
            for field in fields[1:]:
                all_bns = wnkey2bnid.get(field, set())
                all_bns.add(bnid)
                wnkey2bnid[field] = all_bns
    with open(WN_INDEX_SENSE_PATH) as lines, open(out_path, "w") as writer:
        for line in lines:
            fields = line.strip().split(" ")

            if int(fields[2]) == 1:
                sensekey = fields[0]
                lemma = sensekey.split("%")[0]
                pos = get_pos_from_key(sensekey)
                lemmapos = lemma + "#" + pos
                gold = next(iter(wnkey2bnid[sensekey]))
                writer.write("{}\t{}\n".format(lemmapos, gold))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--out_path", required=True)
    args = parser.parse_args()

    build_wn_bnids_mfs(args.out_path + ".en.bnids.txt")
    build_wn_offset_mfs(args.out_path + ".en.wnoffset.txt")
    build_wn_sensekey_mfs(args.out_path + ".en.sensekey.txt")
