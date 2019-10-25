class Lemma2Synsets(dict):
    def __init__(self, path, separator="\t", key_transform=lambda x: x, value_transform=lambda x: x):
        """
        :param path: path to lemma 2 synset map.
        """
        super().__init__()
        with open(path) as lines:
            for line in lines:
                fields = line.strip().split(separator)
                key = key_transform(fields[0])
                synsets = self.get(key, list())
                synsets.extend([value_transform(v) for v in fields[1:]])
                self[key] = synsets