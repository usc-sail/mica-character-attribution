#!/bin/bash

salloc \
--account=shrikann_35 \
--time=2-00:00:00 \
--nodes=1 \
--ntasks=1 \
--partition=epyc-64