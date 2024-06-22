#!/bin/bash

PIPENN_HOME=~/pipenn-exp
JOB_DIR=$PIPENN_HOME/jobs

ALG_DIR=ensnet-ppi
ALG_FILE=ensnet-ppi-keras.py
JOB_DURATION=0:30:00

source $JOB_DIR/common-lisa-job.sh $ALG_DIR $ALG_FILE $JOB_DURATION
