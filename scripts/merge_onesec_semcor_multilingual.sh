#!/usr/bin/env bash
## MERGE SemCor_de + babelnet_glosses_de
export PATH=/home/tommaso/anaconda3/bin:$PATH
export PYTHONPATH=.
echo "OneSeC_de + SemCor_de"
python src/data/data_preparation.py \
merge \
--dataset_paths data/training_data/multilingual_training_data/translated_semcor/de/semcor.de.data.xml \
data/training_data/multilingual_training_data/onesec/de/onesec.de.new_testset_instances.data.xml \
--lang de \
--dataset_name onesec_de+semcor_de \
--outpath data/training_data/multilingual_training_data/onesec+translated_semcor/de/onesec_de+semcor_de.filter.data.xml

## MERGE SemCor_es + babelnet_glosses_es
echo "OneSeC_es + SemCor_es"
python src/data/data_preparation.py \
merge \
--dataset_paths data/training_data/multilingual_training_data/translated_semcor/es/semcor.es.data.xml \
data/training_data/multilingual_training_data/onesec/es/onesec.es.new_testset_instances.data.xml \
--lang es \
--dataset_name onesec_es+semcor_es \
--outpath data/training_data/multilingual_training_data/onesec+translated_semcor/es/onesec_es+semcor_es.filter.data.xml

## MERGE SemCor_fr + babelnet_glosses_fr
echo "OneSeC_fr + SemCor_fr"
python src/data/data_preparation.py \
merge \
--dataset_paths data/training_data/multilingual_training_data/translated_semcor/fr/semcor.fr.data.xml \
data/training_data/multilingual_training_data/onesec/fr/onesec.fr.new_testset_instances.data.xml \
--lang fr \
--dataset_name onesec_fr+semcor_fr \
--outpath data/training_data/multilingual_training_data/onesec+translated_semcor/fr/onesec_fr+semcor_fr.filter.data.xml

## MERGE SemCor_it + babelnet_glosses_it
echo "OneSeC_it + SemCor_it"
python src/data/data_preparation.py \
merge \
--dataset_paths data/training_data/multilingual_training_data/translated_semcor/it/semcor.it.data.xml \
data/training_data/multilingual_training_data/onesec/it/onesec.it.new_testset_instances.data.xml \
--lang it \
--dataset_name onesec_it+semcor_it \
--outpath data/training_data/multilingual_training_data/onesec+translated_semcor/it/onesec_it+semcor_it.filter.data.xml

echo "OneSeC_en + SemCor_en"
python src/data/data_preparation.py \
merge \
--dataset_paths data/training_data/multilingual_training_data/translated_semcor/en/semcor.en.data.xml \
data/training_data/multilingual_training_data/onesec/en/onesec.en.new_testset_instances.data.xml \
--lang en \
--dataset_name onesec_en+semcor_en \
--outpath data/training_data/multilingual_training_data/onesec+translated_semcor/en/onesec_en+semcor_en.filter.data.xml