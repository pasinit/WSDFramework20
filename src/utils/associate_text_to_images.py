import os
import h5py
import pickle as pkl
from tqdm import tqdm


def create_txt_id_2_asscoaited_img_ids(dataset_h5_path, image_dataset_h5_path, ranking_pkl_path,
                                       out_path):
    with open(ranking_pkl_path, "rb") as reader:
        ranking = pkl.load(reader)
    txt_dataset = h5py.File(dataset_h5_path, "r")
    img_dataset = h5py.File(image_dataset_h5_path, "r")

    with open(out_path, "w") as writer:
        for i_ranked, i_sid in tqdm(zip(ranking, txt_dataset["sentence_idxs"]), total=len(ranking)):
            img_names = list()
            for dictionary in i_ranked:
                ranked_img_idx, ranked_img_score = dictionary["corpus_id"], dictionary["score"]
                fname = img_dataset["image_fnames"][ranked_img_idx][0]
                img_names.append(fname + ":" + str(ranked_img_score))
            writer.write(i_sid[0] + "\t" + "\t".join(img_names) + "\n")


MODEL_MAPPING = {
    "sbert_baseline": "sbert-xlm-r-distilroberta-base-paraphrase-v1",
    "sbert_finetuned_bn_glosses": "sbert-saved_models_en_glosses_triplet_loss"}
if __name__ == "__main__":
    txt_dataset_name = "semcor"
    img_dataset_name = "mscoco"
    model_name = "sbert_baseline"
    model_iacer_name = MODEL_MAPPING[model_name]
    root = "/home/tommaso/dev/PycharmProjects/multimodal_wsd_bianca/data/in/new_data_iacer/"
    txt_h5_path = "txt_datasets/{}/{}.data.xml.v1.json.{}.features.h5".format(model_name,
                                                                              txt_dataset_name, model_iacer_name)
    img_h5_path = "img_datasets/{}/Train_{}-training.tsv.{}.features.h5".format(
        model_name, img_dataset_name.upper(), model_iacer_name)
    ranking_pkl_path = "retrieval_results/{}/{}.retrieval-results.{}-queries.{}-corpus.pkl".format(
        model_name, model_iacer_name.replace("sbert-", ""), txt_dataset_name, img_dataset_name)
    out_path = "txt_img_mapping/{}/{}_{}.txt".format(
        model_name, txt_dataset_name, img_dataset_name)
    txt_h5_path = os.path.join(root, txt_h5_path)
    img_h5_path = os.path.join(root, img_h5_path)
    ranking_pkl_path = os.path.join(root, ranking_pkl_path)
    out_path = os.path.join(root, out_path)
    create_txt_id_2_asscoaited_img_ids(
        txt_h5_path, img_h5_path, ranking_pkl_path, out_path)
