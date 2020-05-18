import shutil
import os
def move(path):
    wn_path = path.replace(".txt", ".wn.txt")
    shutil.move(path, wn_path)

def convert(path, out_path, wnoffsets2bn):
    with open(path) as lines, open(out_path, "w") as writer:
        for line in lines:
            fields = line.strip().split()
            bns = list()
            for field in fields[1:]:
                wnid = field.replace("-", "")
                if wnid not in wnoffsets2bn:
                    print(wnid, " not in mapping (", path, ")")
                    continue
                bnid = wnoffsets2bn[wnid]
                bns.append(bnid)
            if len(bns) > 0:
                writer.write(fields[0] + " " + " ".join(bns) + "\n")

def convert_dirs(dirs, wnoffsets2bn):
    for dir in dirs:
        # path = os.path.join(dir, dir.split("/")[-1] + ".gold.key.txt")
        # move(path)
        path = os.path.join(dir, dir.split("/")[-1] + ".gold.key.wn.txt")
        convert(path, path.replace(".wn", ""),  wnoffsets2bn)


if __name__ == "__main__":
    wnoffset2bn = dict()
    with open("/home/tommaso/dev/PycharmProjects/WSDframework/resources/mappings/all_bn_wn.txt") as lines:
        for line in lines:
            fields = line.strip().split("\t")
            for f in fields[1:]:
                wnoffset2bn[f] = fields[0]

    root = "/home/tommaso/Documents/data/WSD_Evaluation_Framework_3.0/Evaluation_Datasets"
    dirs = list()
    for d in ["wordnet-slovenian", "wordnet-galician"]: #wordnet-basque wordnet-bulgarian wordnet-catalan wordnet-chinesesimply "\
              # "wordnet-croatian wordnet-danish wordnet-dutch wordnet-estonian wordnet-estonian wordnet-hungarian wordnet-italian "\
              # "wordnet-japanese wordnet-korean wordnet-spanish".split():
        dirs.append(os.path.join(root, d))

    convert_dirs(dirs, wnoffset2bn)

