#!/usr/bin/env bash
## MERGE SemCor + onesec_en
echo "SemCor + onesec_en"
python src/data/data_preparation.py \
merge \
--dataset_paths /home/tommaso/Documents/data/WSD_Evaluation_Framework/Training_Corpora/SemCor/semcor.data.xml data/onesec+semcor/onesec_en_to_add.data.xml \
--lang en \
--dataset_name semcor+onesec_en \
--outpath data/onesec+semcor/semcor+onesec_en.data.xml

python src/data/data_preparation.py \
convert_key2bn \
--input_key data/onesec+semcor/semcor+onesec_en.gold.key.txt \
--output_key data/onesec+semcor/semcor+onesec_en.gold.key.bn40.txt

## MERGE SemCor + onesec_en + wordnet_gloss_manual
echo "SemCor + onesec_en +  gloss_manual"
python src/data/data_preparation.py \
merge \
--dataset_paths /home/tommaso/Documents/data/WSD_Evaluation_Framework/Training_Corpora/SemCor/semcor.data.xml data/onesec+semcor/onesec_en_to_add.data.xml data/princeton_tagged_glosses/semeval2013_format/princeton_glosses.manual.data.xml \
--lang en \
--dataset_name semcor+onesec_en+glosses_manual \
--outpath data/onesec+semcor+glosses/semcor+onesec_en+glosses_manual.data.xml

python src/data/data_preparation.py \
convert_key2bn \
--input_key data/onesec+semcor+glosses/semcor+onesec_en+glosses_manual.gold.key.txt \
--output_key data/onesec+semcor+glosses/semcor+onesec_en+glosses_manual.gold.key.bn40.txt

## MERGE SemCor + onesec_en + wordnet_gloss_all
echo "SemCor + onesec_en + gloss_all"
python src/data/data_preparation.py \
merge \
--dataset_paths /home/tommaso/Documents/data/WSD_Evaluation_Framework/Training_Corpora/SemCor/semcor.data.xml data/onesec+semcor/onesec_en_to_add.data.xml data/princeton_tagged_glosses/semeval2013_format/princeton_glosses.all.data.xml \
--lang en \
--dataset_name semcor+onesec_en+glosses_all \
--outpath data/onesec+semcor+glosses/semcor+onesec_en+glosses_all.data.xml

python src/data/data_preparation.py \
convert_key2bn \
--input_key data/onesec+semcor+glosses/semcor+onesec_en+glosses_all.gold.key.txt \
--output_key data/onesec+semcor+glosses/semcor+onesec_en+glosses_all.gold.key.bn40.txt
