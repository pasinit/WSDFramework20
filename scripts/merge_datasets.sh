#!/usr/bin/env bash
## MERGE SemCor + princeton_glosses_man
echo "SemCor + princeton_glosses_man"
python src/data/data_preparation.py \
--dataset_paths /home/tommaso/Documents/data/WSD_Evaluation_Framework/Training_Corpora/SemCor/semcor.data.xml data/princeton_tagged_glosses/semeval2013_format/princeton_glosses.manual.data.xml \
--lang en \
--dataset_name semcor+princeton_glosses_man \
--outpath data/training_data/semcor+princeton_glosses_manual.data.xml

## MERGE SemCor + princeton_glosses_all
echo "SemCor + princeton_glosses_all"
python src/data/data_preparation.py \
--dataset_paths /home/tommaso/Documents/data/WSD_Evaluation_Framework/Training_Corpora/SemCor/semcor.data.xml data/princeton_tagged_glosses/semeval2013_format/princeton_glosses.all.data.xml \
--lang en \
--dataset_name semcor+princeton_glosses_man \
--outpath data/training_data/semcor+princeton_glosses_all.data.xml

## MERGE SemCor + princeton_examples_all
echo "SemCor + princeton_examples_all"
python src/data/data_preparation.py \eeeeeeeeeeeeeeeeeeeeeeeeeeeeew
--dataset_paths /home/tommaso/Documents/data/WSD_Evaluation_Framework/Training_Corpora/SemCor/semcor.data.xml data/princeton_tagged_glosses/semeval2013_format/princeton_examples.all.data.xml \
--lang en \
--dataset_name semcor+princeton_glosses_man \
--outpath data/training_data/semcor+princeton_examples_all.data.xml

## MERGE SemCor + princeton_glosses_manual + princeton_examples_all
echo "SemCor + princeton_glosses_manual + princeton_examples_all"
python src/data/data_preparation.py \
--dataset_paths /home/tommaso/Documents/data/WSD_Evaluation_Framework/Training_Corpora/SemCor/semcor.data.xml data/princeton_tagged_glosses/semeval2013_format/princeton_glosses.manual.data.xml data/princeton_tagged_glosses/semeval2013_format/princeton_glosses.manual.data.xml data/princeton_tagged_glosses/semeval2013_format/princeton_examples.all.data.xml \
--lang en \
--dataset_name semcor+princeton_glosses_man \
--outpath data/training_data/semcor+princeton_glosses_manual+princeton_examples_all.data.xml

## MERGE SemCor + princeton_glosses_all + princeton_examples_all
echo "SemCor + princeton_glosses_all + princeton_examples_all"
python src/data/data_preparation.py \
--dataset_paths /home/tommaso/Documents/data/WSD_Evaluation_Framework/Training_Corpora/SemCor/semcor.data.xml data/princeton_tagged_glosses/semeval2013_format/princeton_glosses.manual.data.xml data/princeton_tagged_glosses/semeval2013_format/princeton_glosses.all.data.xml data/princeton_tagged_glosses/semeval2013_format/princeton_examples.all.data.xml \
--lang en \
--dataset_name semcor+princeton_glosses_man \
--outpath data/training_data/semcor+princeton_glosses_all+princeton_examples_all.data.xml