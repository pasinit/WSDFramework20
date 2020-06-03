from lxml import etree


def load_golds(path, ids):
    golds = list()
    with open(path) as lines:
        for line in lines:
            fields = line.strip().split(" ")
            if fields[0] in ids:
                golds.append(line.strip())
    return golds


def sample_xml(path, outpath, size):
    root = etree.parse(path).getroot()
    for t in list(filter(lambda x: len([y for y in x]) < 6, root.findall("./text/sentence"))):
        t.getparent().remove(t)

    for t in root.findall("./text/sentence")[size+1:]:
        t.getparent().remove(t)
    x = root.findall("./text/sentence")[0]
    x.getparent().remove(x)
    ids = set([x.attrib["id"] for x in root.findall("./text/sentence/instance")])
    golds = load_golds(path.replace(".data.xml", ".gold.key.txt"), ids)
    et = etree.ElementTree(root)
    et.write(outpath, pretty_print=True, xml_declaration=True, encoding='UTF-8')
    print(len(ids))
    with open(outpath.replace(".data.xml", ".gold.key.txt"), "w") as writer:
        writer.write("\n".join(golds))


import os

if __name__ == "__main__":
    # path = "/home/tommaso/Documents/data/WSD_Evaluation_Framework_3.0/new_evaluation_datasets"
    # outpath = "/home/tommaso/Documents/data/WSD_Evaluation_Framework_3.0/sample_to_submit"
    # path = "/home/tommaso/dev/PycharmProjects/WSDframework/data2/training_data/multilingual_training_data/TranslatedTrainESCAPED"
    # outpath = "/home/tommaso/dev/PycharmProjects/WSDframework/data2/training_data/sampled_translated_training"
    path = "/home/tommaso/dev/PycharmProjects/WSDframework/data2/training_data/en_training_data/aux/"
    outpath = "/home/tommaso/dev/PycharmProjects/WSDframework/data2/training_data/sampled_training"
    # path = "/home/tommaso/dev/PycharmProjects/WSDframework/data5"
    # outpath = "/home/tommaso/dev/PycharmProjects/WSDframework/sample_en/"

    size = 100
    for d in os.listdir(path):
        print(d)
        if d == "complete_datasets":
            continue
        l_path = os.path.join(path, d, d + ".data.xml")
        out_d = d.replace("_michele", "")

        l_outpath = os.path.join(outpath, out_d, out_d + ".data.xml")
        if not os.path.exists(os.path.join(outpath, out_d)):
            os.makedirs(os.path.join(outpath, out_d))

        sample_xml(l_path, l_outpath,size)
