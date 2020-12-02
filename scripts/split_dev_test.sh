#!/usr/bin/env bash
PYTHONPATH=.
for dataset in `ls /home/tommaso/Documents/data/WSD_Evaluation_Framework_3.0/new_evaluation_datasets/complete_datasets/`
do
    echo $dataset
    language=`echo $dataset | cut -d "-" -f 2`
    python src/data/dataset_dev_test_splitter.py \
        --xml_path /home/tommaso/Documents/data/WSD_Evaluation_Framework_3.0/new_evaluation_datasets/complete_datasets/$dataset/$dataset.data.xml \
        --test_perc 80 \
        --language $language \
        --out_dir /home/tommaso/Documents/data/WSD_Evaluation_Framework_3.0/new_evaluation_datasets/
done