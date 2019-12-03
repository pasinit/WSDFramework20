for filename in Path(".").rglob("wiki_*.data.xml"):
    print(filename)
    for _ in etree.iterparse(str(filename)):
        pass
