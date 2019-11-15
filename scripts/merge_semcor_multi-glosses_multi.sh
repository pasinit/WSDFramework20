#!/usr/bin/env bash
## MERGE SemCor_de + babelnet_glosses_de
echo "SemCor_de + babelnet_glosses_de"
export PATH=/home/tommaso/anaconda3/bin:$PATH
python src/data/data_preparation.py \
merge \
--dataset_paths data/training_data/translated_semcor/semcor.de.data.xml data/training_data/babelnet_multilingual_glosses/glosses_de.parsed.filter.data.xml5 \
--lang de \
--dataset_name semcor_de+babelgloss_de \
--outpath data/training_data/semcor_multilingual+glosses_multilingual/semcor_de+babelgloss_de.filter.data.xml

## MERGE SemCor_es + babelnet_glosses_es
echo "SemCor_es + babelnet_glosses_es"
python src/data/data_preparation.py \
merge \
--dataset_paths data/training_data/translated_semcor/semcor.es.data.xml data/training_data/babelnet_multilingual_glosses/glosses_es.parsed.filter.data.xml5 \
--lang es \
--dataset_name semcor_es+babelgloss_es \
--outpath data/training_data/semcor_multilingual+glosses_multilingual/semcor_es+babelgloss_es.filter.data.xml

## MERGE SemCor_fr + babelnet_glosses_fr
echo "SemCor_fr + babelnet_glosses_fr"
python src/data/data_preparation.py \
merge \
--dataset_paths data/training_data/translated_semcor/semcor.fr.data.xml data/training_data/babelnet_multilingual_glosses/glosses_fr.parsed.filter.data.xml5 \
--lang fr \
--dataset_name semcor_fr+babelgloss_fr \
--outpath data/training_data/semcor_multilingual+glosses_multilingual/semcor_fr+babelgloss_fr.filter.data.xml
