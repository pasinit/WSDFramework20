#!/usr/bin/env bash
export PATH=/home/tommaso/anaconda3/bin:$PATH
export PYTHONPATH=.
echo `pwd`
#echo "MERGING ONESEC + SEMCOR"
#./scripts/merge_onesec_semcor_multilingual.sh
#echo "MERGING ONESEC + GLOSSES"
#./scripts/merge_onesec_glosses_multilingual.sh
#echo "MERGING SEMCOR + GLOSSES"
#./scripts/merge_semcor_glosses_multilingual.sh
echo "MERGING + ONESEC + SEMCOR + GLOSSES"
./scripts/merge_onesec_semcor_glosses_multilingual.sh