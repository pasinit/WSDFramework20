import os
from xml.etree import ElementTree as ET

def check(dirs):
    for dir in dirs:
        path = os.path.join(dir, dir.split("/")[-1] + ".data.xml")
        try:
            ET.parse(path).getroot()
        except Exception as e:
            print(dir, "FAILED")
            print(e)
        print(dir, "OK")


if __name__ == "__main__":
    root = "/home/tommaso/Documents/data/WSD_Evaluation_Framework_3.0/Evaluation_Datasets"
    dirs = list()
    for d in "wordnet-basque wordnet-bulgarian wordnet-catalan wordnet-croatian wordnet-dutch wordnet-hungarian wordnet-italian wordnet-japanese wordnet-spanish".split():
        dirs.append(os.path.join(root, d))
    check(dirs)
