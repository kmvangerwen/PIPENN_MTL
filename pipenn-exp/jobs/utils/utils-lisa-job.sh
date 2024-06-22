#!/bin/bash

PIPENN_HOME=~/pipenn-exp
JOB_DIR=$PIPENN_HOME/jobs

ALG_DIR=utils
ALG_FILE=PPIDataset.py
JOB_DURATION=10:29:00

source $JOB_DIR/common-lisa-job.sh $ALG_DIR $ALG_FILE $JOB_DURATION

