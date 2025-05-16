#!/bin/bash

NSLICES=$1
for (( SLICE=0; SLICE<$NSLICES; SLICE+=4 )); do
    echo $SLICE $NSLICES
    sbatch extract-batch.sh $SLICE $NSLICES
done