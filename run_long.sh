#!/bin/bash

#   Run as a background task
#
#   >>> nohup ./run_long.sh > out/res.log &

GRID="coarse" # or "fine" (too long)
#GRID="fine"
BASES="cc-pvdz unc-cc-pvdz unc-cc-pvtz pc-1 unc-pc-1 pc-2 unc-pc-2 \
    cc-pvtz aug-cc-pvdz aug-cc-pvtz aug-cc-pvqz cc-pvqz cc-pv5z \
    aug-cc-pv5z pc-3 pc-4 aug-pc-3 aug-pc-4 unc-pc-4"

for BASIS in $BASES;
do 
    echo $BASIS
    #python3 examples/estimate.py $BASIS $GRID
    python3 temp.py $BASIS

    #git add out/*
    #git commit -m "ljll server results on $BASIS (fine grid)"
    #git push
done


