from collections import Counter

import stanfordnlp
from xml.etree import ElementTree as ET

from tqdm import tqdm
from xml.dom import minidom


def parse_babelnet_glosses2(input_file, output_file, language):
    all_structured_lines = tokenize_glosses_and_merge_annotations(input_file, language)

    root = ET.Element("corpus")
    root.attrib.update({"lang": language, "source": "babelnet-glosses-%s" % language})

    key2gold = dict()
    for doc_id, structured_glosses in all_structured_lines.items():
        document_xml = ET.SubElement(root, "text")
        document_xml.attrib.update({"id": doc_id})
        for sentence_id, (structured_tokens, source) in enumerate(structured_glosses):
            sentence_xml = ET.SubElement(document_xml, "sentence")
            sentence_xml.attrib["id"] = "%s.s%03d" % (doc_id, sentence_id)
            for token_idx, word, lemma, pos, annotation in structured_tokens:
                if annotation is None:
                    token_xml = ET.SubElement(sentence_xml, "wf")
                else:
                    token_xml = ET.SubElement(sentence_xml, "instance")
                    token_xml.attrib["id"] = doc_id + ".s%03d.t%03d" % (sentence_id, token_idx)
                    key2gold[token_xml] = token_xml.attrib["id"]
                token_xml.attrib.update({"lemma": lemma, "pos": pos})
                token_xml.text = word
    tree = ET.ElementTree(root)
    xmlstr = minidom.parseString(ET.tostring(root)).toprettyxml(indent="   ")
    with open(output_file, "w") as f:
        f.write(xmlstr)


def tokenize_glosses_and_merge_annotations(input_file, language):
    pipeline = stanfordnlp.Pipeline(processors="tokenize,pos,lemma", use_gpu=True, lang=language)
    all_structured_lines = dict()
    with open(input_file) as lines:
        counter = 0
        for line in tqdm(lines):
            fields = line.strip().split("\t")
            doc_id = fields[0]
            gloss = fields[1]
            annotations = fields[2:-1]
            source = fields[-1]
            annotation_dict = dict()
            for annotation in annotations:
                annotation_fields = annotation.split(":")
                token = ":".join(annotation_fields[0:-4])
                start, end = [int(x) for x in annotation_fields[-2:]]
                bnid = ":".join(annotation_fields[-4:-2])
                annotation_dict[token] = {"start": start, "end": end, "bnid": bnid}
            annotations = sorted(annotation_dict.items(), key=lambda elem: elem[1]["start"])
            i = 0
            while i < len(annotations) - 1:  # merge annotations that are contained one in another.
                token, a = annotations[i]
                if a["end"] > annotations[i + 1][1]["end"]:
                    del annotations[i + 1]
                    continue
                elif a["start"] == annotations[i + 1][1]["start"] and a["end"] < annotations[i + 1][1]["end"]:
                    del annotations[i]
                    continue
                i += 1
            tokens = list()
            last_start = 0
            for annotation_token, annotation in annotations:
                to_parse = gloss[last_start:annotation["start"]].strip()
                if len(to_parse) == 0:
                    sentences = []
                else:
                    sentences = pipeline(to_parse).sentences
                for sentence in sentences:
                    for t in sentence.tokens:
                        t_words = t.words
                        wlta = [(tw.text, tw.lemma, tw.upos, None) for tw in t_words]
                        tokens.extend(wlta)
                aux = annotation_token.replace(" ", "_")
                tokens.append((aux, aux, "NOUN", annotation))
                last_start = annotation["end"] + 1
            if last_start < len(gloss):
                to_parse = gloss[last_start:].strip()
                if len(to_parse) == 0:
                    parsed_toks = []
                else:
                    sentences = pipeline(to_parse).sentences
                for sentence in sentences:
                    for t in sentence.tokens:
                        t_words = t.words
                        wlta = [(tw.text, tw.lemma, tw.upos, None) for tw in t_words]
                        tokens.extend(wlta)

            indexed_merged_tokens = list()
            for i, mt in enumerate(tokens):
                indexed_merged_tokens.append(tuple([i] + list(mt)))

            l = all_structured_lines.get(doc_id, list())
            l.append((indexed_merged_tokens, source))
            all_structured_lines[doc_id] = l
            counter += 1
            # if counter >= 10:
            #     break
    return all_structured_lines


if __name__ == "__main__":
    parse_babelnet_glosses2("/home/tommaso/dev/eclipseWorkspace/factories/output/framework20/glosses_it.txt",
                            "/home/tommaso/dev/eclipseWorkspace/factories/output/framework20/glosses_it.parsed.txt",
                            "it")
