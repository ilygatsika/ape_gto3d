#!/bin/bash

<<<<<<< HEAD
#   >>> nohup ./example/estimator_01.sh > out/estimator_01/out.log &
=======
#   >>> nohup ./example/estimator_01 > out/estimator_01/out.log &
>>>>>>> e724d646a150cccb7dc63a70aaeccb7fe9245990

OUT_DIR=out/estimator_01 
#BASES="cc-pvdz unc-cc-pvdz unc-cc-pvtz pc-1 unc-pc-1 pc-2 unc-pc-2 \
#    cc-pvtz aug-cc-pvdz aug-cc-pvtz aug-cc-pvqz cc-pvqz cc-pv5z \
#    aug-cc-pv5z pc-3 pc-4 aug-pc-3 aug-pc-4 unc-pc-4"
BASES="cc-pvdz unc-cc-pvdz unc-cc-pvtz pc-1"

for BASIS in $BASES;
do 
<<<<<<< HEAD
    python3 main.py $BASIS $OUT_DIR
=======
    $PYTHON main.py $BASIS $OUT_DIR
>>>>>>> e724d646a150cccb7dc63a70aaeccb7fe9245990
done

git add out/estimator_01
git commit -m "ljll server results for estimator_01 (fine grid)"
git push

