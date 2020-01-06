echo "IT"
python src/test/evaluate_model.py --config config/config_it_onesec.yaml --checkpoint data/models/it_onesec/bert-base-multilingual-cased/checkpoints/ --output_path data/models/it_onesec/bert-base-multilingual-cased/evaluation/ --find_best --dev_set ~/Documents/data/WSD_Evaluation_Framework_2.0/Evaluation_Datasets/wiki_dev_it_wnfilter/wiki_dev_it_wnfilter.data.xml
echo "ES"
python src/test/evaluate_model.py --config config/config_es_onesec.yaml --checkpoint data/models/es_onesec/bert-base-multilingual-cased/checkpoints/ --output_path data/models/es_onesec/bert-base-multilingual-cased/evaluation/ --find_best --dev_set ~/Documents/data/WSD_Evaluation_Framework_2.0/Evaluation_Datasets/wiki_dev_es_wnfilter/wiki_dev_es_wnfilter.data.xml
echo "FR"
python src/test/evaluate_model.py --config config/config_fr_onesec.yaml --checkpoint data/models/fr_onesec/bert-base-multilingual-cased/checkpoints/ --output_path data/models/fr_onesec/bert-base-multilingual-cased/evaluation/ --find_best --dev_set ~/Documents/data/WSD_Evaluation_Framework_2.0/Evaluation_Datasets/wiki_dev_fr_wnfilter/wiki_dev_fr_wnfilter.data.xml
echo "DE"
python src/test/evaluate_model.py --config config/config_de_onesec.yaml --checkpoint data/models/de_onesec/bert-base-multilingual-cased/checkpoints/ --output_path data/models/de_onesec/bert-base-multilingual-cased/evaluation/ --find_best --dev_set ~/Documents/data/WSD_Evaluation_Framework_2.0/Evaluation_Datasets/wiki_dev_de_wnfilter/wiki_dev_de_wnfilter.data.xml
