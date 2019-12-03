import os

def print_mfs_info(non_mfs_predictions):
    mfs_predictions = non_mfs_predictions.replace(".txt", ".mfs.txt")
    outpath = non_mfs_predictions.replace(".txt", ".mfs.info.txt")
    mfs_ids = set()
    with open(non_mfs_predictions) as lines:
        for line in lines:
            fields = line.strip().split(" ")
            if "<unk>" == fields[-1]:
                mfs_ids.add(fields[0])
    with open(mfs_predictions) as lines, open(outpath, "w") as writer:
        for line in lines:
            fields = line.strip().split(" ")
            if fields[0] in mfs_ids:
                writer.write(line.strip() + " MFS\n")
            else:
                writer.write(line)

def print_mfs_info_by_folder(folder):
    for f in os.listdir(folder):
        if f.endswith(".predictions.txt"):
            print_mfs_info(os.path.join(folder, f))
print_mfs_info_by_folder("data/models/en_semcor_sensekeys_mfs/bert-base-cased/evaluation/")
# print_mfs_info("data/models/en_semcor_gloss_manual_bert_large/bert-large-cased/evaluation/ALL.data.xml.predictions.txt")
exit(0)


input_xml = "/media/tommaso/4940d845-c3f3-4f0b-8985-f91a0b453b07/WSDframework/data/onesec+semcor/semcor.integr.words.2.1.700.all.lexical.data.xml"
output_xml = "data/onesec+semcor/onesec_en_to_add.data.xml"
output_gold_key = "data/onesec+semcor/onesec_en_to_add.gold.key.txt"
corpus_name = "onesec"
with open(input_xml) as reader:
    root = etree.parse(reader).getroot()
corpus = etree.Element("corpus", {"lang": root.attrib["lang"], "source": corpus_name})
docid2textxml = dict()
sentence_counter = 0
with open(output_gold_key, "w") as writer:
    for instance in root.findall("./lexelt/instance"):
        if "docsrc" not in instance.attrib:
            continue
        docid = instance.attrib["docsrc"].replace(" ", "_")
        src = "onesec"
        gold_answer = instance.find("answer").attrib["senseid"]
        context_xml = instance.find("context")
        lemmapos = gold_answer.split(":")[0]
        lemma = lemmapos.split("%")[0]
        pos = get_pos_from_key(gold_answer)
        head_xml = context_xml[0]
        text_xml = docid2textxml.get(docid, etree.SubElement(corpus, "text", {"id": docid, "source": src}))
        text_pre = context_xml.text.strip().split(" ")
        head_text = head_xml.text.strip()
        text_post = head_xml.tail.strip().split(" ")

        sentence_xml = etree.SubElement(text_xml, "sentence", {"id": str(sentence_counter)})
        for tok in text_pre:
            wf = etree.SubElement(sentence_xml, "wf", dict(lemma="", pos=""))
            wf.text = tok
        instance = etree.SubElement(sentence_xml, "instance", dict(lemma=lemma, pos=get_universal_pos(pos),
                                                                   id=docid + "." + "%03d" % sentence_counter + ".000"))
        writer.write(docid + "." + "%03d" % sentence_counter + ".000 " + gold_answer + "\n")
        instance.text = head_text
        for tok in text_post:
            wf = etree.SubElement(sentence_xml, "wf", dict(lemma="", pos=""))
            wf.text = tok
        sentence_counter += 1

et = etree.ElementTree(corpus)
et.write(output_xml, pretty_print=True, xml_declaration=True, encoding='UTF-8')

exit(0)

root = ET.parse("/home/tommaso/Documents/data/SemEval/SemEval2013/data/multilingual-all-words.de.xml").getroot()
sentences = dict()
tok2word = dict()
for sentence in root.findall("text/sentence"):
    s = ""
    sid = sentence.attrib["id"]
    for token in sentence:
        if "id" in token.attrib:
            tok2word[token.attrib["id"]] = token.text
        s += token.text + " "

    sentences[sid] = s.strip()

with open("/home/tommaso/Telegram Desktop/babelnet.13.de.tocheck.key.bn40.changed") as reader, \
        open("/home/tommaso/Telegram Desktop/babelnet.13.de.tocheck.key.bn40.changed.with_sentences.tsv",
             "w") as writer:
    for line in reader:
        fields = line.strip().split("\t")
        tid = fields[0]
        sid = ".".join(fields[0].split(".")[:2])
        word = tok2word[tid]
        sentence = sentences[sid]
        if tid == "d006.s007.t005":
            print()
        writer.write(line.strip() + "\t" + word + "\t" + sentence.strip() + "\n")
