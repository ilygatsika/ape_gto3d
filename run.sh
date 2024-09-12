#!/bin/bash

#   >>> nohup ./example/estimator_01.sh > out/estimator_01/out.log &
# User-defined arguments
# grid = coarse (fast), fine (slow, use with caution)

helpFunction()
{
   echo ""
   echo "Usage: $0 -g option"
   echo -e "\t-g Specifies grid option=fine (slow, use with caution) or option=coarse (fast)"
   exit 1
}

while getopts "g:" opt
do
   case "$opt" in
      g ) grid="$OPTARG" ;;
      ? ) helpFunction   ;;
   esac
done

# Print helpFunction in case parameters are empty
if [ -z "$grid" ]
then
   echo "A parameter is empty";
   helpFunction
fi

echo "$grid"

BASES="cc-pvdz unc-cc-pvdz unc-cc-pvtz pc-1 unc-pc-1 pc-2 unc-pc-2 \
    cc-pvtz aug-cc-pvdz aug-cc-pvtz aug-cc-pvqz cc-pvqz cc-pv5z \
    aug-cc-pv5z pc-3 pc-4 aug-pc-3 aug-pc-4 unc-pc-4"
#BASES="cc-pvdz unc-cc-pvdz unc-cc-pvtz pc-1"
OUT_DIR=out/estimator_01/res.pickle 

# TODO deal with grid option, then density file
IN_WFK=density.chk
IN_DEN=helfem.chk

IN_WFK=density_small.chk
IN_DEN=helfem_small.chk

if [ -e $OUT_DIR ]
then
    echo "Loads precomputation"
else
    echo "No precomputation found. Starts computation"

    for BASIS in $BASES;
        do 
          echo $BASIS
          python3 main.py $BASIS $OUT_DIR $IN_WFK $IN_DEN

          #git add out/estimator_01
          #git commit -m "ljll server results on $BASIS (fine grid)"
          #git push
        done
    fi
  fi


