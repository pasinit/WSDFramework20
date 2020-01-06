#!/usr/bin/env bash
## MERGE SemCor_de + babelnet_glosses_de
export PATH=/home/tommaso/anaconda3/bin:$PATH
export PYTHONPATH=.
echo "OneSeC_de + babelnet_glosses_de"
python src/data/data_preparation.py \
merge \
--dataset_paths data/training_data/multilingual_training_data/babel_glosses_train_dev/de/glosses_de.filter.train.data.xml \
data/training_data/multilingual_training_data/onesec/de/onesec.de.new_testset_instances.data.xml \
--lang de \
--dataset_name onesec_de+babelgloss_de \
--outpath data/training_data/multilingual_training_data/onesec+babel_glosses/de/onesec_de+babelgloss_de.filter.data.xml

## MERGE SemCor_es + babelnet_glosses_es
echo "OneSeC_es + babelnet_glosses_es"
python src/data/data_preparation.py \
merge \
--dataset_paths data/training_data/multilingual_training_data/babel_glosses_train_dev/es/glosses_es.filter.train.data.xml \
data/training_data/multilingual_training_data/onesec/es/onesec.es.new_testset_instances.data.xml \
--lang es \
--dataset_name onesec_es+babelgloss_es \
--outpath data/training_data/multilingual_training_data/onesec+babel_glosses/es/onesec_es+babelgloss_es.filter.data.xml

## MERGE SemCor_fr + babelnet_glosses_fr
echo "OneSeC_fr + babelnet_glosses_fr"
python src/data/data_preparation.py \
merge \
--dataset_paths data/training_data/multilingual_training_data/babel_glosses_train_dev/fr/glosses_fr.filter.train.data.xml \
data/training_data/multilingual_training_data/onesec/fr/onesec.fr.new_testset_instances.data.xml \
--lang fr \
--dataset_name onesec_fr+babelgloss_fr \
--outpath data/training_data/multilingual_training_data/onesec+babel_glosses/fr/onesec_fr+babelgloss_fr.filter.data.xml

## MERGE SemCor_it + babelnet_glosses_it
echo "OneSeC_it + babelnet_glosses_it"
python src/data/data_preparation.py \
merge \
--dataset_paths data/training_data/multilingual_training_data/babel_glosses_train_dev/it/glosses_it.filter.train.data.xml \
data/training_data/multilingual_training_data/onesec/it/onesec.it.new_testset_instances.data.xml \
--lang it \
--dataset_name onesec_it+babelgloss_it \
--outpath data/training_data/multilingual_training_data/onesec+babel_glosses/it/onesec_it+babelgloss_it.filter.data.xml

echo "OneSeC_en + babelnet_glosses_en"
python src/data/data_preparation.py \
merge \
--dataset_paths data/training_data/multilingual_training_data/babel_glosses_train_dev/en/glosses_en.filter.train.data.xml \
data/training_data/multilingual_training_data/onesec/en/onesec.en.new_testset_instances.data.xml \
--lang en \
--dataset_name onesec_en+babelgloss_en \
--outpath data/training_data/multilingual_training_data/onesec+babel_glosses/en/onesec_en+babelgloss_en.data.xml