#!/bin/bash

#   >>> nohup ./example/estimator_01 > out/estimator_01/out.log &

OUT_DIR=out/estimator_01 
#BASES="cc-pvdz unc-cc-pvdz unc-cc-pvtz pc-1 unc-pc-1 pc-2 unc-pc-2 \
#    cc-pvtz aug-cc-pvdz aug-cc-pvtz aug-cc-pvqz cc-pvqz cc-pv5z \
#    aug-cc-pv5z pc-3 pc-4 aug-pc-3 aug-pc-4 unc-pc-4"
BASES="cc-pvdz unc-cc-pvdz unc-cc-pvtz pc-1"

for BASIS in $BASES;
do 
    $PYTHON main.py $BASIS $OUT_DIR
done

git add out/estimator_01
git commit -m "ljll server results for estimator_01 (fine grid)"
git push

