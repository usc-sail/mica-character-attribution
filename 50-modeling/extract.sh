#!/bin/bash

NSLICES=$1
for (( SLICE=0; SLICE<$NSLICES; SLICE+=1 )); do
    sbatch extract-batch.sh $SLICE $NSLICES
done