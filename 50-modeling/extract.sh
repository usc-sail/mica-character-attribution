#!/bin/bash

NPROCESSES=$1
for (( PROCESS=0; PROCESS<$NPROCESSES; PROCESS+=2 )); do
    sbatch extract-batch.sh $PROCESS $NPROCESSES
done