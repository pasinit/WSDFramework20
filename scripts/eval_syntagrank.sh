#!/usr/bin/env bash

cd /home/tommaso/Documents/data/WSD_Evaluation_Framework/Evaluation_Datasets/
syntagrank_predicitons=$1
datasets=`echo $2 | tr ',' ' '`
for dataset in $datasets
do
#    echo /home/tommaso/Documents/data/WSD_Evaluation_Framework_3.0/Evaluation_Datasets/$dataset/$dataset.gold.key.txt
#    echo $syntagrank_predicitons/$dataset.predictions.txt
     echo $dataset `java Scorer \
        /home/tommaso/Documents/data/WSD_Evaluation_Framework_3.0/new_evaluation_datasets/$dataset/$dataset.gold.key.txt \
        $syntagrank_predicitons/$dataset.predictions.txt | grep "F1" | sed -r "s/F1=\s+//g"`
done