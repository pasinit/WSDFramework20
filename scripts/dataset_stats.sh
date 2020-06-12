VERSION=0.1.6
SPLIT=all

cd ..
BASE_DIR=`/home/tommaso/Documents/data/WSD_Evaluation_Framework_3.0/new_evaluation_datasets/`
PACKAGE_DIR=data/out/releases
PACKAGE_NAME=multilingual_wsd_${SPLIT}_v$VERSION
PACKAGE_ROOT=$PACKAGE_DIR/$PACKAGE_NAME

cd $PACKAGE_ROOT
echo `pwd`
echo "instances:"
wc -l semeval2013-*/semeval2013-*.gold.key.txt | grep -v "total"
wc -l semeval2015-*/semeval2015-*.gold.key.txt | grep -v "total"

cd $BASE_DIR
echo "Word Types:"
for dataset in semeval2013-it semeval2013-es semeval2013-fr semeval2013-de semeval2015-it semeval2015-es
do
  python scripts/count_word_types.py $PACKAGE_ROOT/$dataset/$dataset.data.xml
done

echo "Unique BN Synsets"
cd $PACKAGE_ROOT
for dataset in semeval2013-it semeval2013-es semeval2013-fr semeval2013-de semeval2015-it semeval2015-es
do
  echo $dataset `cat $dataset/$dataset.gold.key.txt | cut -d " " -f 1 --complement | tr " " "\n" | sort | uniq | wc -l`
done

cd $BASE_DIR
echo "Polysemy:"
for dataset in semeval2013-it semeval2013-es semeval2013-fr semeval2013-de semeval2015-it semeval2015-es
do
  lang=`echo $dataset | cut -d "-" -f 2`
  python scripts/dataset_polysemy.py $PACKAGE_ROOT/$dataset/$dataset.data.xml $PACKAGE_ROOT/inventories/inventory.$lang.withgold.txt
done