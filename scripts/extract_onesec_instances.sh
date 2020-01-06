#!/usr/bin/env bash

export PYTHONPATH=.
export PATH=/home/tommaso/anaconda3/bin/:$PATH
LANG=en
UPLANG=EN
python src/data/data_preparation.py \
extract_from_onesec \
--test_set_instances_path data/lemmas_se13-15_bn/lemmaposTestBN.$LANG \
--already_covered_path data/training_data/en_training_data/onesec_original_data/onesec_testset_instances/OneSeC_$UPLANG.data.xml \
--outpath data/training_data/multilingual_training_data/onesec/$LANG/onesec.$LANG.new_testset_instances.data.xml \
--complete_onesec_path data/training_data/en_training_data/onesec_original_data/onesec_testset_instances/OneSeC_EN.data.xml \
--lang $LANG

LANG=es
UPLANG=ES
python src/data/data_preparation.py \
extract_from_onesec \
--test_set_instances_path data/lemmas_se13-15_bn/lemmaposTestBN.$LANG \
--already_covered_path data/training_data/en_training_data/onesec_original_data/onesec_testset_instances/OneSeC_$UPLANG.data.xml \
--outpath data/training_data/multilingual_training_data/onesec/$LANG/onesec.$LANG.new_testset_instances.data.xml \
--complete_onesec_path data/training_data/multilingual_training_data/onesec/$LANG/onesec.$LANG.all.data.xml \
--lang $LANG

LANG=fr
UPLANG=FR
python src/data/data_preparation.py \
extract_from_onesec \
--test_set_instances_path data/lemmas_se13-15_bn/lemmaposTestBN.$LANG \
--already_covered_path data/training_data/en_training_data/onesec_original_data/onesec_testset_instances/OneSeC_$UPLANG.data.xml \
--outpath data/training_data/multilingual_training_data/onesec/$LANG/onesec.$LANG.new_testset_instances.data.xml \
--complete_onesec_path data/training_data/multilingual_training_data/onesec/$LANG/onesec.$LANG.all.data.xml \
--lang $LANG

LANG=de
UPLANG=DE
python src/data/data_preparation.py \
extract_from_onesec \
--test_set_instances_path data/lemmas_se13-15_bn/lemmaposTestBN.$LANG \
--already_covered_path data/training_data/en_training_data/onesec_original_data/onesec_testset_instances/OneSeC_$UPLANG.data.xml \
--outpath data/training_data/multilingual_training_data/onesec/$LANG/onesec.$LANG.new_testset_instances.data.xml \
--complete_onesec_path data/training_data/multilingual_training_data/onesec/$LANG/onesec.$LANG.all.data.xml \
--lang $LANG


LANG=it
UPLANG=IT
python src/data/data_preparation.py \
extract_from_onesec \
--test_set_instances_path data/lemmas_se13-15_bn/lemmaposTestBN.$LANG \
--already_covered_path data/training_data/en_training_data/onesec_original_data/onesec_testset_instances/OneSeC_$UPLANG.data.xml \
--outpath data/training_data/multilingual_training_data/onesec/$LANG/onesec.$LANG.new_testset_instances.data.xml \
--complete_onesec_path data/training_data/multilingual_training_data/onesec/$LANG/onesec.$LANG.all.data.xml \
--lang $LANG

