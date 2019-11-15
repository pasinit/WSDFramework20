#!/usr/bin/env bash
## MERGE SemCor_de + babelnet_glosses_de
echo "OneSeC_de + SemCor_de + babelnet_glosses_de"
export PATH=/home/tommaso/anaconda3/bin:$PATH
export PYTHONPATH=.
python src/data/data_preparation.py \
merge \
--dataset_paths data/training_data/translated_semcor/semcor.de.data.xml \
data/training_data/babelnet_multilingual_glosses/glosses_de.parsed.filter.data.xml5 \
data/training_data/onesec_testset_instances/OneSeC_DE.data.xml \
--lang de \
--dataset_name onesec_de+semcor_de+babelgloss_de \
--outpath data/training_data/onesec_multilingual+semcor_multilingual+glosses_multilingual/onesec_de+semcor_de+babelgloss_de.filter.data.xml

## MERGE SemCor_es + babelnet_glosses_es
echo "OneSeC_es + SemCor_es + babelnet_glosses_es"
python src/data/data_preparation.py \
merge \
--dataset_paths data/training_data/translated_semcor/semcor.es.data.xml \
data/training_data/babelnet_multilingual_glosses/glosses_es.parsed.filter.data.xml5 \
data/training_data/onesec_testset_instances/OneSeC_ES.data.xml \
--lang es \
--dataset_name onesec_es+semcor_es+babelgloss_es \
--outpath data/training_data/onesec_multilingual+semcor_multilingual+glosses_multilingual/onesec_es+semcor_es+babelgloss_es.filter.data.xml

## MERGE SemCor_fr + babelnet_glosses_fr
echo "OneSeC_fr + SemCor_fr + babelnet_glosses_fr"
python src/data/data_preparation.py \
merge \
--dataset_paths data/training_data/translated_semcor/semcor.fr.data.xml \
data/training_data/babelnet_multilingual_glosses/glosses_fr.parsed.filter.data.xml5 \
data/training_data/onesec_testset_instances/OneSeC_FR.data.xml \
--lang fr \
--dataset_name onesec_fr+semcor_fr+babelgloss_fr \
--outpath data/training_data/onesec_multilingual+semcor_multilingual+glosses_multilingual/onesec_fr+semcor_fr+babelgloss_fr.filter.data.xml
