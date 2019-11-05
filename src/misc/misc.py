import xml.etree.ElementTree as ET

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

with open("/home/tommaso/Telegram Desktop/babelnet.13.de.tocheck.key.bn40.changed") as reader,\
    open("/home/tommaso/Telegram Desktop/babelnet.13.de.tocheck.key.bn40.changed.with_sentences.tsv", "w") as writer:
    for line in reader:
        fields = line.strip().split("\t")
        tid = fields[0]
        sid = ".".join(fields[0].split(".")[:2])
        word = tok2word[tid]
        sentence = sentences[sid]
        if tid == "d006.s007.t005":
            print()
        writer.write(line.strip() + "\t" + word + "\t" + sentence.strip() + "\n")


