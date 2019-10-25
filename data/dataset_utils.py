from data.datasets import Vocabulary, WSDXmlInMemoryDataset
import _pickle as pkl

def serialise_dataset(xml_data_path, key_gold_path, vocabulary_path, tokeniser, model_name, out_file,
                      key2bnid_path=None, key2wnid_path=None):
    vocabulary = Vocabulary.vocabulary_from_gold_key_file(vocabulary_path, key2bnid_path=key2bnid_path)

    dataset = WSDXmlInMemoryDataset(xml_data_path, key_gold_path, vocabulary,
                                    device="cuda",
                                    batch_size=32,
                                    key2bnid_path=key2bnid_path,
                                    key2wnid_path=key2wnid_path)

    with open(out_file, "wb") as writer:
        pkl.dump(dataset, writer)
