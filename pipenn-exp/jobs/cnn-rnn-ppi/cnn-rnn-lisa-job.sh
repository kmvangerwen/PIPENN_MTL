#!/bin/bash


PIPENN_HOME=~/pipenn-exp
JOB_DIR=$PIPENN_HOME/jobs

ALG_DIR=cnn-rnn-ppi
ALG_FILE=cnn-rnn-XD-ppi-keras.py
JOB_DURATION=15:29:00

source $JOB_DIR/common-lisa-job.sh $ALG_DIR $ALG_FILE $JOB_DURATION
