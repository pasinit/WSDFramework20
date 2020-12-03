from src.data.multimodal_dataset import read_wn_inventory


def convert(path, outpath):
    *_, sense2offset = read_wn_inventory("/opt/WordNet-3.0/dict/index.sense")
    with open(path) as reader, open(outpath, "w") as writer:
        for line in reader:
            fields = line.strip().split(" ")
            id = fields[0]
            offsets = [sense2offset[x] for x in fields[1:]]
            writer.write(id + " " + " ".join(offsets) + "\n")


if __name__ == "__main__":
    path = "resources/evaluation_framework_3.0/new_evaluation_datasets/dev-en/dev-en.gold.key.txt"
    outpath = "resources/evaluation_framework_3.0/new_evaluation_datasets/dev-en/dev-en.gold.wn-offset-key.txt"
    convert(path, outpath)