from collections import Counter
import os
import stanfordnlp
from tqdm import tqdm
import _pickle as pkl
from lxml import etree
import spacy

from src.data.dataset_utils import get_simplified_pos, get_pos_from_key, get_universal_pos


def parse_babelnet_glosses2(input_file, output_file, language):
    all_structured_lines = tokenize_glosses_and_merge_annotations(input_file, language)

    key2gold = dict()
    with open(output_file, "w") as writer:
        writer.write('<?xml version="1.0" encoding="UTF-8" ?>\n')
        writer.write('<corpus lang="{}" source="babelnet-glosses-{}">\n'.format(language, language))
        for doc_id, structured_glosses in all_structured_lines.items():
            document_xml = etree.Element("text")
            document_xml.attrib.update({"id": doc_id})
            for sentence_id, (structured_tokens, source) in enumerate(structured_glosses):
                sentence_xml = etree.SubElement(document_xml, "sentence")
                sentence_xml.attrib["id"] = "%s.s%03d" % (doc_id, sentence_id)
                for token_idx, word, lemma, pos, annotation in structured_tokens:
                    if annotation is None:
                        token_xml = etree.SubElement(sentence_xml, "wf")
                    else:
                        token_xml = etree.SubElement(sentence_xml, "instance")
                        token_xml.attrib["id"] = doc_id + ".s%03d.t%03d" % (sentence_id, token_idx)
                        key2gold[token_xml.attrib["id"]] = annotation["bnid"]
                    token_xml.attrib.update({"lemma": lemma if lemma is not None else word, "pos": pos})
                    token_xml.text = word
            writer.write(str(etree.tostring(document_xml, pretty_print=True, encoding="utf8"), "utf-8"))
        writer.write("</corpus>\n")
    with open(output_file.replace(".xml", ".gold.key.txt"), "w") as writer:
        for key, gold in key2gold.items():
            writer.write(key + "\t" + gold + "\n")


def get_model_by_language(language):
    if language != "en":
        return "{}_core_news_sm".format(language)
    else:
        return "en_core_web_sm"


def get_pipeline_by_language(language):
    if language == "en":
        return spacy.load(get_model_by_language(language), disable=["parser", "ner"])
    else:
        return stanfordnlp.Pipeline(processors="tokenize,pos,lemma", use_gpu=True, lang=language)


def parse_text(pipeline, to_parse):
    tokens = []
    if type(pipeline) == stanfordnlp.Pipeline:
        sentences = pipeline(to_parse).sentences
        for sentence in sentences:
            for t in sentence.tokens:
                t_words = t.words
                wlta = [(tw.text, tw.lemma, tw.upos, None) for tw in t_words]
                tokens.extend(wlta)
    else:
        doc = pipeline(to_parse)
        for tw in doc:
            tokens.append((tw.text, tw.lemma_, tw.pos_, None))
    return tokens

def tokenize_glosses_and_merge_annotations(input_file, language):
    # stanfordnlp.download(language)
    # pipeline = stanfordnlp.Pipeline(processors="tokenize,pos,lemma", use_gpu=True, lang=language)
    pipeline = get_pipeline_by_language(language)

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
                if len(to_parse) > 0:
                    wlta = parse_text(pipeline, to_parse)
                tokens.extend(wlta)
                aux = annotation_token.replace(" ", "_")
                tokens.append((aux, aux, "NOUN", annotation))
                last_start = annotation["end"] + 1
            if last_start < len(gloss):
                to_parse = gloss[last_start:].strip()
                if len(to_parse) > 0:
                    tagged_text = parse_text(pipeline, to_parse)
                    tokens.extend(tagged_text)

            indexed_merged_tokens = list()
            for i, mt in enumerate(tokens):
                indexed_merged_tokens.append(tuple([i] + list(mt)))

            l = all_structured_lines.get(doc_id, list())
            l.append((indexed_merged_tokens, source))
            all_structured_lines[doc_id] = l
            counter += 1
<<<<<<< HEAD
=======
            # if counter >= 100:
            #     break
>>>>>>> 46b9577940bbb8d6bc404e983da35a4735babd7c
    return all_structured_lines


def get_annotations(parent, wf, token_id, annotation_type):
    golds = list()
    for id_xml in wf:
        if id_xml.attrib["sk"] != 'purposefully_ignored%0:00:00::':
            golds.append(id_xml.attrib["sk"])
    if len(golds) == 0:
        instance = etree.SubElement(parent, "wf", {"pos": ""})
        text = list(wf.itertext())[-1]
        instance.text = text
        instance.attrib["lemma"] = text
        return instance, None
    lemma = wf.attrib["lemma"].split("|")
    if len(lemma) > 1:
        lemma = golds[0].split("%")[0]
    else:
        lemma = lemma[0].split("%")[0]
    instance = etree.SubElement(parent, "instance",
                                {"lemma": lemma,
                                 "pos": get_universal_pos(get_pos_from_key(golds[0])),
                                 "annotation_type": annotation_type,
                                 "id": token_id
                                 })
    instance.text = list(wf.itertext())[-1]
    return instance, golds


def convert_princeton_tagged_glosses_format(xml_root, out_path, sentence_tag="def", valid_annotation_type=None):
    if valid_annotation_type is None:
        valid_annotation_type = {"man"}
    new_root = etree.Element("corpus", {"lang": "en", "source": "princeton-tagged-{}".format(
        "glosses" if sentence_tag == "def" else "examples")})
    key_lines = list()

    for pos in ["noun", "verb", "adj", "adv"]:
        with open(os.path.join(xml_root, pos + ".xml")) as reader:
            root = etree.parse(reader).getroot()
        print(pos)
        for synset in tqdm(root.findall("./synset")):
            num_defs = 0
            offset = "wn:" + synset.attrib["ofs"] + synset.attrib["pos"]
            text = etree.SubElement(new_root, "text", {"id": offset})
            added = False
            for tagged_definition in synset.findall("./gloss/{}".format(sentence_tag)):

                if len(tagged_definition) > 0 and tagged_definition[0].tag == "qf":
                    # print("porco dio")
                    tagged_definition = tagged_definition[0]
                l = [x for x in tagged_definition if "tag" in x.attrib and x.attrib["tag"] in valid_annotation_type]
                if len(l) == 0:
                    continue
                added = True
                gloss_id = offset + ".s%03d" % num_defs
                gloss_xml = etree.SubElement(text, "sentence", {"id": gloss_id})
                num_defs += 1
                ti = 0
                for token in tagged_definition:
                    if "tag" not in token.attrib:
                        continue
                    if token.attrib["tag"] in valid_annotation_type:
                        token_id = gloss_id + ".t%03d" % ti
                        instance, golds = get_annotations(gloss_xml, token, token_id, token.attrib["tag"])
                        if golds is not None:
                            key_lines.append(instance.attrib["id"] + " " + " ".join(golds))
                            ti += 1
                    elif token.tag == "wf":
                        wf: etree.Element = etree.SubElement(gloss_xml, "wf",
                                                             {"lemma": token.attrib.get("lemma", token.text),
                                                              "pos": get_universal_pos(get_simplified_pos(
                                                                  token.attrib[
                                                                      "pos"])) if "pos" in token.attrib else "O"})
                        text_str = token.text
                        if text_str.strip() == "":
                            text_str = list(token.itertext())[-1]
                        wf.text = text_str
            if not added:
                new_root.remove(text)

    et = etree.ElementTree(new_root)
    et.write(out_path, pretty_print=True)
    with open(out_path.replace("data.xml", "gold.key.txt"), "w") as writer:
        writer.write("\n".join(key_lines))


if __name__ == "__main__":
    # convert_princeton_tagged_glosses_format("data/princeton_tagged_glosses/merged/",
    #                                         "data/princeton_tagged_glosses/semeval2013_format/princeton_glosses.manual.data.xml",
    #                                         valid_annotation_type={"man"})
    # convert_princeton_tagged_glosses_format("data/princeton_tagged_glosses/merged/",
    #                                         "data/princeton_tagged_glosses/semeval2013_format/princeton_glosses.all.data.xml",
    #                                         valid_annotation_type={"man", "auto"})
    # convert_princeton_tagged_glosses_format("data/princeton_tagged_glosses/merged/",
    #                                         "data/princeton_tagged_glosses/semeval2013_format/princeton_examples.manual.xml",
    #                                         valid_annotation_type={"man"}, sentence_tag="ex")
    # convert_princeton_tagged_glosses_format("data/princeton_tagged_glosses/merged/",
    #                                         "data/princeton_tagged_glosses/semeval2013_format/princeton_examples.all.data.xml",
    #                                         valid_annotation_type={"man", "auto"}, sentence_tag="ex")

    parse_babelnet_glosses2("data/training_data/babelnet_multilingual_glosses/glosses_it.txt",
                            "data/training_data/babelnet_multilingual_glosses/glosses_it.parsed.xml.test",
                            "it")
    # # lang = "de"
    # # with open("/home/tommaso/dev/eclipseWorkspace/factories/output/framework20/glosses_de.parsed.txt.pkl", "rb") as reader:
    # #     xml = pkl.load(reader)
    # #     print("xml loaded")
    # #     for sentence in xml.findall("text/sentence"):
    # #         for tok in sentence:
    # #             if tok.attrib["lemma"] is None:
    # #                 tok.attrib["lemma"] = tok.text
    # #     xml.attrib["lang"] = lang
    # #     # xml = ET.parse("/home/tommaso/Documents/data/WSD_Evaluation_Framework/Evaluation_Datasets/semeval2015/semeval2015.data.xml")
    # # et = ElementTree.ElementTree(xml)
    # # # for e in et.iter():
    # # #     print()
    # # et.write("/home/tommaso/dev/eclipseWorkspace/factories/output/framework20/glosses_de.parsed.temp.xml")
    # # del et
    # # del xml
    # et = etree.parse("/home/tommaso/dev/eclipseWorkspace/factories/output/framework20/glosses_de.parsed.xml")
    # et.write("/home/tommaso/dev/eclipseWorkspace/factories/output/framework20/glosses_de.parsed.pretty.xml", pretty_print=True)
