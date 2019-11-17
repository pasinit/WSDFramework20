#!/usr/bin/env bash

export PATH=/home/tommaso/anaconda3/bin:$PATH
export PYTHONPATH=.
# MERGE SemCor_de + babelnet_glosses_de
echo "SemCor_de + babelnet_glosses_de"
python src/data/data_preparation.py \
merge \
--dataset_paths data/training_data/translated_semcor/de/semcor.de.data.xml \
data/training_data/babelnet_multilingual_glosses/glosses_de.parsed.filter.data.xml \
--lang de \
--dataset_name semcor_de+babelgloss_de \
--outpath data/training_data/semcor_multilingual+glosses_multilingual/semcor_de+babelgloss_de.filter.data.xml

## MERGE SemCor_es + babelnet_glosses_es
echo "SemCor_es + babelnet_glosses_es"
python src/data/data_preparation.py \
merge \
--dataset_paths data/training_data/translated_semcor/es/semcor.es.data.xml data/training_data/babelnet_multilingual_glosses/glosses_es.parsed.filter.data.xml \
--lang es \
--dataset_name semcor_es+babelgloss_es \
--outpath data/training_data/semcor_multilingual+glosses_multilingual/semcor_es+babelgloss_es.filter.data.xml

## MERGE SemCor_fr + babelnet_glosses_fr
echo "SemCor_fr + babelnet_glosses_fr"
python src/data/data_preparation.py \
merge \
--dataset_paths data/training_data/translated_semcor/fr/semcor.fr.data.xml data/training_data/babelnet_multilingual_glosses/glosses_fr.parsed.filter.data.xml \
--lang fr \
--dataset_name semcor_fr+babelgloss_fr \
--outpath data/training_data/semcor_multilingual+glosses_multilingual/semcor_fr+babelgloss_fr.filter.data.xml

## MERGE SemCor_it + babelnet_glosses_it
echo "SemCor_it + babelnet_glosses_it"
python src/data/data_preparation.py \
merge \
--dataset_paths data/training_data/translated_semcor/it/semcor.it.data.xml data/training_data/babelnet_multilingual_glosses/glosses_it.parsed.filter.data.xml \
--lang it \
--dataset_name semcor_it+babelgloss_it \
--outpath data/training_data/semcor_multilingual+glosses_multilingual/semcor_it+babelgloss_it.filter.data.xml

echo "SemCor_en + babelnet_glosses_en"
python src/data/data_preparation.py \
merge \
--dataset_paths data/training_data/translated_semcor/en/semcor.en.data.xml \
 data/training_data/babelnet_multilingual_glosses/glosses_en.parsed.filter.data.xml \
--lang en \
--dataset_name semcor_en+babelgloss_en \
--outpath data/training_data/semcor_multilingual+glosses_multilingual/semcor_en+babelgloss_en.filter.data.xml