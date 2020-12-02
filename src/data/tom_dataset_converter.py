from lxml import etree
from tqdm import tqdm


def check_duplicate(input_file):
    sentences = set()
    counter = 0
    with open(input_file) as lines:
        for line in tqdm(lines):
            if not line.strip().startswith("<answer"):
                continue
            xml_line = etree.fromstring(line)
            context = next(lines)
            if "</context>" not in context:
                context = "<context>" + next(lines).strip()
            xml_context = etree.fromstring(context)
            head = xml_context[0]
            pre_text = xml_context.text.strip().split(" ") if xml_context.text is not None else []
            pre_text = [token.split("/")[0] for token in pre_text]
            head_text = [head.text.strip().split("/")[0]]
            post_text = head.tail.strip().split(" ")
            post_text = [token.split("/")[0] for token in post_text]
            txt = "".join(pre_text + head_text + post_text)
            if txt in sentences:
                print(txt)
                counter += 1
            else:
                sentences.add(txt)
    print(counter)


def to_framework_xml(input_file, output_file, lang):
    gold_file = output_file.replace(".data.xml", ".gold.key.txt")
    with open(input_file) as lines, open(output_file, "w") as writer, \
            open(gold_file, "w") as gold_writer:
        writer.write("<?xml version='1.0' encoding='utf-8'?>\n")
        writer.write('<corpus lang="{}" source="train-o-matic">\n'.format(lang))
        sentence_id = 0
        text_counter = 0
        last_instance = None
        xml_document = None
        for line in tqdm(lines):
            instance_counter = 0
            if not line.strip().startswith("<answer"):
                continue
            xml_line = etree.fromstring(line)
            instance_id = xml_line.attrib["instance"].split(".")[0] + "_n" + ".{}".format(text_counter)
            if xml_document is None:
                xml_document = etree.Element("text", attrib={"id": str(text_counter)})
                last_instance = instance_id
            else:
                if last_instance != instance_id:
                    writer.write(str(etree.tostring(xml_document, pretty_print=True, encoding="utf8"), "utf-8") + "\n")
                    text_counter += 1
                    xml_document = etree.Element("text", attrib={"id": str(text_counter)})
                    last_instance = instance_id
                    sentence_id = 0
            gold_sense = xml_line.attrib["sensekey"]
            context = next(lines)
            if "</context>" not in context:
                context = "<context>" + next(lines).strip()
            xml_context = etree.fromstring(context)
            xml_out_sentence = etree.Element("sentence", attrib={"id": "{}.{}".format(text_counter, sentence_id)})
            if len(xml_context) == 0:
                continue
            head = xml_context[0]
            pre_text = xml_context.text.strip().split(" ") if xml_context.text is not None else ""
            head_text = head.text.strip().split("/")
            post_text = head.tail.strip().split(" ")
            for token in pre_text:
                fields = token.split("/")
                if fields[0] == "":
                    continue
                attribs = dict()
                if len(fields) == 3:
                    attribs = {"lemma": fields[1], "pos": fields[2]}
                wf = etree.Element("wf", attrib=attribs)
                wf.text = fields[0]
                xml_out_sentence.append(wf)
            if len(head_text) == 1:
                lemma = gold_sense.split("%")[0]
                pos = "NN"
            else:
                lemma = head_text[1]
                pos = head_text[-1]
            xml_instance = etree.Element("instance",
                                         attrib={"id": "{}.{}.{}".format(str(text_counter), sentence_id, instance_counter),
                                                 "lemma": lemma,
                                                 "pos": pos
                                                 }
                                         )
            xml_instance.text = head_text[0]
            xml_out_sentence.append(xml_instance)
            for token in post_text:
                fields = token.split("/")
                if fields[0] == "":
                    continue
                attribs = dict()
                if len(fields) == 3:
                    attribs = {"lemma": fields[1], "pos": fields[2]}
                wf = etree.Element("wf", attrib=attribs)
                wf.text = fields[0]
                xml_out_sentence.append(wf)

            xml_document.append(xml_out_sentence)
            gold_writer.write("{}.{}.{}\t{}\n".format(text_counter, sentence_id, instance_counter, gold_sense))
            instance_counter += 1
            sentence_id += 1
        writer.write("</corpus>")


def merge_sentences(xml_path):
    gold_path = xml_path.replace("data.xml", "gold.key.txt")
    id2gold = dict()
    with open(gold_path, "rt") as lines:
        for line in lines:
            fields = line.strip().split("\t")
            id2gold[fields[0]] = fields[1]
    gold_writer = open(gold_path.replace("gold.key.txt", "new.gold.key.txt"), "wt")
    data = etree.parse(xml_path)
    sentence2xml = dict()
    sentences = data.findall("./text/sentence")
    for sentence in tqdm(sentences):
        txt = "".join([token.text for token in sentence])
        prev_xml = sentence2xml.get(txt)
        if prev_xml is not None:
            old_doc_id = [x for x in prev_xml if x.tag == "instance"][0].attrib["id"]
            old_doc_id = ".".join(old_doc_id.split(".")[:2])
            instance_idx, new_instances = [x for x in enumerate(sentence) if x[1].tag == "instance"][0]
            old_token = prev_xml[instance_idx]
            old_token.tag = "instance"
            old_token.attrib["id"] = old_doc_id + "." + ".".join(new_instances.attrib["id"].split(".")[2:])
            old_token.attrib["gold"] = id2gold[new_instances.attrib["id"]]
            sentence.getparent().remove(sentence)
        else:
            sentence2xml[txt] = sentence
    for sentence in data.findall("./text/sentence"):
        instances = sentence.findall("./instance")
        if len(instances) == 1:
            continue
        for idx, instance in enumerate(instances):
            old_id = instance.attrib["id"]
            if "gold" in instance.attrib:
                gold = instance.attrib["gold"]
                del instance.attrib["gold"]
            else:
                gold = id2gold[old_id]
            new_id = ".".join(old_id.split(".")[0:-1]) + ".{}".format(idx)
            gold_writer.write(new_id + "\t" + gold + "\n")
            instance.attrib["id"] = new_id
    data.write(xml_path.replace("data.xml", "new.data.xml"), pretty_print=True, xml_declaration=True, encoding='UTF-8')
    gold_writer.close()


if __name__ == "__main__":
    for lang in ["EN"]:#["IT", "ES", "FR", "DE", "ZH"]:
        print(lang)
        to_framework_xml(
            "/media/tommaso/My Book/train-o-matic/{}/evaluation-framework-ims-training.utf8.xml".format(lang),
            "/media/tommaso/My Book/train-o-matic/{}/evaluation-framework-ims-training.fw20.data.xml".format(
                lang),
            lang.lower())
        merge_sentences("/media/tommaso/My Book/train-o-matic/{}/evaluation-framework-ims-training.fw20.data.xml".format(lang))
