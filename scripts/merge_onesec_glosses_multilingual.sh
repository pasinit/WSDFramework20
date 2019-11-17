#!/usr/bin/env bash
## MERGE SemCor_de + babelnet_glosses_de
echo "OneSeC_de + babelnet_glosses_de"
export PATH=/home/tommaso/anaconda3/bin:$PATH
export PYTHONPATH=.
python src/data/data_preparation.py \
merge \
--dataset_paths data/training_data/babelnet_multilingual_glosses/glosses_de.parsed.filter.data.xml \
data/training_data/onesec/de/onesec.de.new_testset_instances.data.xml \
--lang de \
--dataset_name onesec_de+babelgloss_de \
--outpath data/training_data/onesec_multilingual+glosses_multilingual/onesec_de+babelgloss_de.filter.data.xml

## MERGE SemCor_es + babelnet_glosses_es
echo "OneSeC_es + babelnet_glosses_es"
python src/data/data_preparation.py \
merge \
--dataset_paths data/training_data/babelnet_multilingual_glosses/glosses_es.parsed.filter.data.xml \
data/training_data/onesec/es/onesec.es.new_testset_instances.data.xml \
--lang es \
--dataset_name onesec_es+babelgloss_es \
--outpath data/training_data/onesec_multilingual+glosses_multilingual/onesec_es+babelgloss_es.filter.data.xml

## MERGE SemCor_fr + babelnet_glosses_fr
echo "OneSeC_fr + babelnet_glosses_fr"
python src/data/data_preparation.py \
merge \
--dataset_paths data/training_data/babelnet_multilingual_glosses/glosses_fr.parsed.filter.data.xml \
data/training_data/onesec/fr/onesec.fr.new_testset_instances.data.xml \
--lang fr \
--dataset_name onesec_fr+babelgloss_fr \
--outpath data/training_data/onesec_multilingual+glosses_multilingual/onesec_fr+babelgloss_fr.filter.data.xml

## MERGE SemCor_it + babelnet_glosses_it
echo "OneSeC_it + babelnet_glosses_it"
python src/data/data_preparation.py \
merge \
--dataset_paths data/training_data/babelnet_multilingual_glosses/glosses_it.parsed.filter.data.xml \
data/training_data/onesec/it/onesec.it.new_testset_instances.data.xml \
--lang it \
--dataset_name onesec_it+babelgloss_it \
--outpath data/training_data/onesec_multilingual+glosses_multilingual/onesec_it+babelgloss_it.filter.data.xml