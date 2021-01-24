import h5py
from xml.etree.ElementTree import ElementTree as ET
def debug(semcor_path, sentence2imgname_path, imgname2caption_path):
    imgname2caption = dict()
    with open(imgname2caption_path) as lines:
        for line in lines:
            imgname, caption = line.strip().split("\t")
            imgname2caption[imgname] = caption
    sentenceid2name = dict()
    with open(sentence2imgname_path) as lines:
        for line in lines:
            imgname, sentenceid = line.strip().split("\t")
            sentenceid2name[sentenceid] = imgname
    root = ET.parse(semcor_path).getroot()
    for sentence in root.findall("./text/sentence"):
        input("press enter")
        words = [token.text for token in sentence]
        sentenceid = sentence.attr["id"]
        imgname = sentenceid2name[sentenceid]
        caption = imgname2caption[imgname]
        print(sentenceid, " ".join(words))
        print(imgname, caption)
        ## todo visualize image from google conceptual caption folder
    