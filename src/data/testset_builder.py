import lxml.etree as etree
import os


def filter_instances_without_wn(data_xml_path, out_dir):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    gold_path = data_xml_path.replace(".data.xml", ".gold.key.txt")
    out_gold_path = os.path.join(out_dir, gold_path.split("/")[-1].replace(".gold.key.txt", "_wnfilter.gold.key.txt"))
    out_xml_path = os.path.join(out_dir, data_xml_path.split("/")[-1].replace(".data.xml", "_wnfilter.data.xml"))
    ids_to_keep = set()
    with open(gold_path) as lines, open(out_gold_path, "w") as writer:
        for line in lines:
            fields = line.strip().split(" ")
            iid = fields[0]
            bnids = [x for x in fields[1:] if int(x[3:-1]) < 117660 or x == "bn:14866890n"]
            if len(bnids) > 0:
                writer.write(line)
                ids_to_keep.add(iid)
    xml_root = etree.parse(data_xml_path).getroot()
    for instance in xml_root.findall(".text/sentence/instance"):
        if instance.attrib["id"] in ids_to_keep:
            continue
        instance.tag = "wf"
        del instance.attrib["id"]
    et = etree.ElementTree(xml_root)
    et.write(out_xml_path, pretty_print=True)


if __name__ == "__main__":
    for lang in ["it", "es", "fr", "de"]:
        data_xml_path = "/home/tommaso/Documents/data/WSD_Evaluation_Framework_2.0/Evaluation_Datasets/semeval2013_{}/semeval2013_{}.data.xml".format(
            lang, lang)
        out_dir = "/home/tommaso/Documents/data/WSD_Evaluation_Framework_2.0/Evaluation_Datasets/semeval2013_{}_wnfilter/".format(
            lang)
        filter_instances_without_wn(data_xml_path, out_dir)
        data_xml_path = "/home/tommaso/Documents/data/WSD_Evaluation_Framework_2.0/Evaluation_Datasets/wiki_dev_{}/wiki_dev_{}.data.xml".format(
            lang, lang)
        out_dir = "/home/tommaso/Documents/data/WSD_Evaluation_Framework_2.0/Evaluation_Datasets/wiki_dev_{}_wnfilter/".format(
            lang)
        filter_instances_without_wn(data_xml_path, out_dir)
        data_xml_path = "/home/tommaso/Documents/data/WSD_Evaluation_Framework_2.0/Evaluation_Datasets/wiki_test_{}/wiki_test_{}.data.xml".format(
            lang, lang)
        out_dir = "/home/tommaso/Documents/data/WSD_Evaluation_Framework_2.0/Evaluation_Datasets/wiki_test_{}_wnfilter/".format(
            lang)
        filter_instances_without_wn(data_xml_path, out_dir)

    for lang in ["it", "es"]:
        data_xml_path = "/home/tommaso/Documents/data/WSD_Evaluation_Framework_2.0/Evaluation_Datasets/semeval2015_{}/semeval2015_{}.data.xml".format(
            lang, lang)
        out_dir = "/home/tommaso/Documents/data/WSD_Evaluation_Framework_2.0/Evaluation_Datasets/semeval2015_{}_wnfilter/".format(
            lang)
        filter_instances_without_wn(data_xml_path, out_dir)


