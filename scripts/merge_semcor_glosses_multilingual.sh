#!/usr/bin/env bash

export PATH=/home/tommaso/anaconda3/bin:$PATH
export PYTHONPATH=.
echo "SemCor_de + babelnet_glosses_de"
python src/data/data_preparation.py \
merge \
--dataset_paths data/training_data/multilingual_training_data/translated_semcor/de/semcor.de.data.xml \
data/training_data/multilingual_training_data/babel_glosses_train_dev/de/glosses_de.filter.train.data.xml \
--lang de \
--dataset_name semcor_de+babelgloss_de \
--outpath data/training_data/multilingual_training_data/translated_semcor+babel_glosses/de/semcor_de+babelgloss_de.filter.data.xml

## MERGE SemCor_es + babelnet_glosses_es
echo "SemCor_es + babelnet_glosses_es"
python src/data/data_preparation.py \
merge \
--dataset_paths data/training_data/multilingual_training_data/translated_semcor/es/semcor.es.data.xml \
data/training_data/multilingual_training_data/babel_glosses_train_dev/es/glosses_es.filter.train.data.xml \
--lang es \
--dataset_name semcor_es+babelgloss_es \
--outpath data/training_data/multilingual_training_data/translated_semcor+babel_glosses/es/semcor_es+babelgloss_es.filter.data.xml

# MERGE SemCor_fr + babelnet_glosses_fr
echo "SemCor_fr + babelnet_glosses_fr"
python src/data/data_preparation.py \
merge \
--dataset_paths data/training_data/multilingual_training_data/translated_semcor/fr/semcor.fr.data.xml \
data/training_data/multilingual_training_data/babel_glosses_train_dev/fr/glosses_fr.filter.train.data.xml \
--lang fr \
--dataset_name semcor_fr+babelgloss_fr \
--outpath data/training_data/multilingual_training_data/translated_semcor+babel_glosses/fr/semcor_fr+babelgloss_fr.filter.data.xml

## MERGE SemCor_it + babelnet_glosses_it
echo "SemCor_it + babelnet_glosses_it"
python src/data/data_preparation.py \
merge \
--dataset_paths data/training_data/multilingual_training_data/translated_semcor/it/semcor.it.data.xml \
data/training_data/multilingual_training_data/babel_glosses_train_dev/it/glosses_it.filter.train.data.xml \
--lang it \
--dataset_name semcor_it+babelgloss_it \
--outpath data/training_data/multilingual_training_data/translated_semcor+babel_glosses/it/semcor_it+babelgloss_it.filter.data.xml

echo "SemCor_en + babelnet_glosses_en"
python src/data/data_preparation.py \
merge \
--dataset_paths data/training_data/multilingual_training_data/translated_semcor/en/semcor.en.data.xml \
data/training_data/multilingual_training_data/babel_glosses_train_dev/en/glosses_en.filter.train.data.xml \
--lang en \
--dataset_name semcor_en+babelgloss_en \
--outpath data/training_data/multilingual_training_data/translated_semcor+babel_glosses/en/semcor_en+babelgloss_en.filter.data.xml