#!/bin/bash

PIPENN_HOME=~/pipenn-exp
JOB_DIR=$PIPENN_HOME/jobs

ALG_DIR=dnet-ppi
ALG_FILE=dnet-XD-ppi-keras.py
#JOB_DURATION=72:29:00
JOB_DURATION=25:29:00

source $JOB_DIR/common-lisa-job.sh $ALG_DIR $ALG_FILE $JOB_DURATION