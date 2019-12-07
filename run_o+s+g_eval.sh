echo "IT"
python src/test/evaluate_model.py --config config/config_it_s+g+o.yaml --checkpoint data/models/it_s+g+o/bert-base-multilingual-cased/checkpoints/ --output_path data/models/it_s+g+o/bert-base-multilingual-cased/evaluation/ --find_best --dev_set ~/Documents/data/WSD_Evaluation_Framework_2.0/Evaluation_Datasets/wiki_dev_it_wnfilter/wiki_dev_it_wnfilter.data.xml

echo "ES"
python src/test/evaluate_model.py --config config/config_es_s+g+o.yaml --checkpoint_path data/models/es_s+g+o/bert-base-multilingual-cased/checkpoints/ --output_path data/models/es_s+g+o/bert-base-multilingual-cased/evaluation/ --dev_set ~/Documents/data/WSD_Evaluation_Framework_2.0/Evaluation_Datasets/wiki_dev_es/wiki_dev_es.data.xml --find_bes

echo "FR"
python src/test/evaluate_model.py --config config/config_fr_s+g+o.yaml --checkpoint_path data/models/fr_s+g+o/bert-base-multilingual-cased/checkpoints/ --output_path data/models/fr_s+g+o/bert-base-multilingual-cased/evaluation/ --dev_set ~/Documents/data/WSD_Evaluation_Framework_2.0/Evaluation_Datasets/wiki_dev_fr/wiki_dev_fr.data.xml --find_best

echo "DE"
python src/test/evaluate_model.py --config config/config_de_s+g+o.yaml --checkpoint_path data/models/de_s+g+o/bert-base-multilingual-cased/checkpoints/ --output_path data/models/de_s+g+o/bert-base-multilingual-cased/evaluation/ --dev_set ~/Documents/data/WSD_Evaluation_Framework_2.0/Evaluation_Datasets/wiki_dev_de/wiki_dev_de.data.xml --find_best
