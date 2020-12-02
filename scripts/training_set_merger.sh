#!/usr/bin/env bash


BASEDIR=/home/tommaso/dev/PycharmProjects/WSDframework/data2/training_data/multilingual_training_data/TranslatedTrainESCAPED
OUTDIR=/home/tommaso/dev/PycharmProjects/WSDframework/data2/training_data/multilingual_training_data/TranslatedTrainESCAPED_merged
#for LANG in bg ca da de es et eu fr gl hr hu it ja nl sl
#do
#    echo $LANG
#    mkdir $OUTDIR/train-$LANG
#    python src/data/dataset_merger.py --xml_paths $BASEDIR/semcor_$LANG/semcor_$LANG.data.xml \
#        $BASEDIR/wngt_michele_glosses_$LANG/wngt_michele_glosses_$LANG.data.xml \
#        $BASEDIR/wngt_michele_examples_$LANG/wngt_michele_examples_$LANG.data.xml \
#    --language $LANG \
#    --output_dir $OUTDIR/train-$LANG
#done

BASEDIR=/home/tommaso/dev/PycharmProjects/WSDframework/data2/training_data/en_training_data
OUTDIR=/media/tommaso/4940d845-c3f3-4f0b-8985-f91a0b453b07/WSDframework/data/training_data/en_training_data/semcor+michele_wngt
LANG=en
python src/data/dataset_merger.py --xml_paths $BASEDIR/semcor/semcor.data.xml \
        $BASEDIR/wngt_michele/wngt_michele_examples/wngt_michele_examples.data.xml \
        $BASEDIR/wngt_michele/wngt_michele_glosses/wngt_michele_glosses.data.xml \
    --language $LANG \
    --output_dir $OUTDIR/train-$LANG