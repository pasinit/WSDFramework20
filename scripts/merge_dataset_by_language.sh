#!/usr/bin/env bash
PYTHONPATH=.
echo "IT"
python src/data/dataset_merger.py \
--xml_paths \
/home/tommaso/Documents/data/WSD_Evaluation_Framework_3.0/Evaluation_Datasets/semeval2010-it/semeval2010-it.data.xml \
/home/tommaso/Documents/data/WSD_Evaluation_Framework_3.0/Evaluation_Datasets/semeval2013-it/semeval2013-it.data.xml \
/home/tommaso/Documents/data/WSD_Evaluation_Framework_3.0/Evaluation_Datasets/semeval2015-it/semeval2015-it.data.xml \
--language it \
--output_dir /home/tommaso/Documents/data/WSD_Evaluation_Framework_3.0/new_evaluation_datasets/wsd-it

echo "ES"
python src/data/dataset_merger.py \
--xml_paths \
/home/tommaso/Documents/data/WSD_Evaluation_Framework_3.0/Evaluation_Datasets/semeval2013-es/semeval2013-es.data.xml \
/home/tommaso/Documents/data/WSD_Evaluation_Framework_3.0/Evaluation_Datasets/semeval2015-es/semeval2015-es.data.xml \
--language es \
--output_dir /home/tommaso/Documents/data/WSD_Evaluation_Framework_3.0/new_evaluation_datasets/wsd-es

echo "FR"
python src/data/dataset_merger.py \
--xml_paths \
/home/tommaso/Documents/data/WSD_Evaluation_Framework_3.0/Evaluation_Datasets/semeval2013-fr/semeval2013-fr.data.xml \
--language fr \
--output_dir /home/tommaso/Documents/data/WSD_Evaluation_Framework_3.0/new_evaluation_datasets/wsd-fr

echo "DE"
python src/data/dataset_merger.py \
--xml_paths \
/home/tommaso/Documents/data/WSD_Evaluation_Framework_3.0/Evaluation_Datasets/semeval2013-de/semeval2013-de.data.xml \
--language de \
--output_dir /home/tommaso/Documents/data/WSD_Evaluation_Framework_3.0/new_evaluation_datasets/wsd-de


echo "ZH"
python src/data/dataset_merger.py \
--xml_paths \
/home/tommaso/Documents/data/WSD_Evaluation_Framework_3.0/Evaluation_Datasets/semeval2010-zh/semeval2010-zh.data.xml \
/home/tommaso/Documents/data/WSD_Evaluation_Framework_3.0/Evaluation_Datasets/wordnet-chinesesimply/wordnet-chinesesimply.data.xml \
--language zh \
--output_dir /home/tommaso/Documents/data/WSD_Evaluation_Framework_3.0/new_evaluation_datasets/wsd-zh

echo "EN"
python src/data/dataset_merger.py \
--xml_paths \
/home/tommaso/Documents/data/WSD_Evaluation_Framework_3.0/Evaluation_Datasets/senseval2/senseval2.data.xml \
/home/tommaso/Documents/data/WSD_Evaluation_Framework_3.0/Evaluation_Datasets/senseval3/senseval3.data.xml \
/home/tommaso/Documents/data/WSD_Evaluation_Framework_3.0/Evaluation_Datasets/semeval2010/semeval2010.data.xml \
/home/tommaso/Documents/data/WSD_Evaluation_Framework_3.0/Evaluation_Datasets/semeval2013/semeval2013.data.xml \
/home/tommaso/Documents/data/WSD_Evaluation_Framework_3.0/Evaluation_Datasets/semeval2015/semeval2015.data.xml \
--language en \
--output_dir /home/tommaso/Documents/data/WSD_Evaluation_Framework_3.0/new_evaluation_datasets/wsd-en



for aux in basque,eu bulgarian,bg catalan,ca croatian,hr danish,da dutch,nl estonian,et galician,gl hungarian,hu japanese,ja korean,ko slovenian,sl
do

    language=`echo $aux | cut -d ',' -f 1`
    lang_code=`echo $aux | cut -d ',' -f 2`
    echo "wordnet-$language"
    python src/data/dataset_merger.py \
    --xml_paths \
    /home/tommaso/Documents/data/WSD_Evaluation_Framework_3.0/Evaluation_Datasets/wordnet-$language/wordnet-$language.data.xml \
    --language $lang_code \
    --output_dir /home/tommaso/Documents/data/WSD_Evaluation_Framework_3.0/new_evaluation_datasets/wsd-$lang_code
done