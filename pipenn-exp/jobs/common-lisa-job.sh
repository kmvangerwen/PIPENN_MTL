#!/bin/bash


## As it is not possible to pass parameters to SBATCH, we create this SBATCH script dynamically. Thereafter we pass the whole generated script to the
## sbatch command. 

cat << EOF  | sbatch
#!/bin/bash

###########################################
##SBATCH -p defq
##SBATCH --cpus-per-task=32

##SBATCH -p binf
##SBATCH --cpus-per-task=32

##SBATCH -p bw
##SBATCH --cpus-per-task=20

# uncomment this one if yoy to use gpu; Note that it shouldn't be used with pytorch for generating embedding (doesn't work because of some bug).
##SBATCH --gres=gpu:2
###########################################

# change partion if you want
##SBATCH -p defq
#SBATCH -p binf
#SBATCH --cpus-per-task=32
##SBATCH --gres=gpu:2

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=$3

#SBATCH --job-name=$1
#SBATCH --output=$1_%j.out

#module load cudnn8.1-cuda11.2/8.1.1.33 
#module load 2022; module load cuDNN/8.4.1.50-CUDA-11.7.0
#module load 2023; module load cuDNN/8.7.0.84-CUDA-11.8.0
# Cuda has been installed with conda tensorflow-gpu. We need to put the required libraries of Cunda (libdevice.10.bc) in the path.
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

# This has already been set by specific alg-jobs.
#PIPENN_HOME=~/pipenn-exp

CONDA_HOME=/scistor/informatica/rhu300/miniconda3
VENV_HOME=/scistor/informatica/rhu300/venv-ws/reza-tfg-2.12
JOB_DIR=\$PIPENN_HOME/jobs
#PIPENN_ENV=pipenn-1.0
#PIPENN_ENV=rezac-tfg-2.12
PIPENN_ENV=reza-tfg-2.14

ALG_DIR=$1
ALG_FILE=$2
SLURM_OUT=\$SLURM_SUBMIT_DIR/$1_\$SLURM_JOB_ID.out 

prepare_run() {
	source \$CONDA_HOME/etc/profile.d/conda.sh
  	conda deactivate
  	conda deactivate
        export PIPENN_HOME=~/pipenn-exp/
  	export PYTHONPATH=\$PIPENN_HOME/utils:\$PIPENN_HOME/config:\$PIPENN_HOME/ann-ppi:\$PIPENN_HOME/rnn-ppi:\$PIPENN_HOME/dnet-ppi:\$PIPENN_HOME/rnet-ppi:\$PIPENN_HOME/unet-ppi:\$PIPENN_HOME/cnn-rnn-ppi:\$PIPENN_HOME/ensnet-ppi:\$PIPENN_HOME:\$PIPENN_HOME/mtnet-ppi

	# by uncommenting this command, you can use conda, instead of python virtual env
        echo "PIPENN_ENV is: \$PIPENN_ENV"
  	conda activate \$PIPENN_ENV

	# by uncommenting these commands, you can use python virtual env, instead of  conda
	#deactivate
	#source \$VENV_HOME/bin/activate
}

# Use this for PIPENN (experimental) and not for USERDS (webservice)
# Note: Students need to define CONDA_HOME in their .bash_profile as follows: CONDA_HOME=/scistor/informatica/rhu300/miniconda3
gen_pipenn_preds() {
	python \$PIPENN_HOME/\$ALG_DIR/\$ALG_FILE
}

# Given a user-data-set (containing protein sequences in FASTA format), we generate interface predictions for each protein.
# Put your file (needs to be called "user_input.fasta") in a folder (jout), uncomment this, and start from jobs/alg_dir/alg_job 
# (jobs/unet-ppi//unet-lisa-job.sh). The generated files will be placed in the same folder (jout).
# Note: model file needs to be in  /scistor/informatica/rhu300/pipenn-exp/models/all-models/dataset*/model*
# Example: /scistor/informatica/rhu300/pipenn-exp/models/all-models/biolip-p/dnet-ppi-model.hdf5
gen_userds_preds() {
	export PRED_TYPE='p'
	export USERDS_INPUT_DIR=.
	export USERDS_INPUT_FASTA_FILE=\$USERDS_INPUT_DIR/user_input.fasta
	export USERDS_OUTPUT_FILE=\$USERDS_INPUT_DIR/prepared_userds.csv
	export UTIL_DIR=utils
	export MFG_FILE=GenerateMinFeatures.py
	mkdir ./\$ALG_DIR

	# Use this one to generate a table from a fasta file and also to generate protbert embeddings for each protein in the fasta file.
	# Fasta file must contain two lines for each protein: (1) starts with a line ">id " and (2) the sequence. 
	python \$PIPENN_HOME/\$UTIL_DIR/\$MFG_FILE \$USERDS_INPUT_FASTA_FILE \$USERDS_OUTPUT_FILE

	python \$PIPENN_HOME/\$ALG_DIR/\$ALG_FILE \$PIPENN_HOME \$PRED_TYPE
}

gen_all_preds() {
	export SPLIT_DIR="mhc-bp-test/split_files"
	files=("\$SPLIT_DIR"/*)
	for file in "\${files[@]}"; do
  	  cp \$file prepared_userds.csv
	  echo "### File copied: \$file"

	  gen_userds_preds
	  #wait
	  if [[ \$? -ne 0 ]]; then
            echo "### Python script failed. Exiting loop."
            exit 1
    	  fi

	  export NUM=\$(echo "\$file" | sed 's/[^0-9]*\([0-9]\+\).*/\1/')
	  export PRED_DIR="mhc-bp-test/pred_files/slice_\$NUM"
	  #echo "PRED_DIR is \$PRED_DIR"
	  mkdir \$PRED_DIR
	  mv ./\$ALG_DIR/* \$PRED_DIR
	  mv prepared_userds.csv \$PRED_DIR
	  echo "Preds generated for \$file"
	done

}

do_computation() {
  	echo "preparing computation at $(date) ..."
  	prepare_run
  	echo "executing prediction at $(date) ..."
	
	# Use this for PIPENN (experimental)
  	gen_pipenn_preds

	# Use this for PIPENN web-service (Embedding version)
  	#gen_userds_preds

	# We split a big fasta file to a number of smaller files (each max 500 proteins). Use this command to generate predictions for splitted files. 
  	#gen_all_preds
}
do_computation &
wait

EOF
