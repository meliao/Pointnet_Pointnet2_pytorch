#!/bin/bash
#SBATCH --job-name=pointnet2_001
#SBATCH --partition=contrib-gpu
#SBATCH --output=logs/001.out
#SBATCH --error=logs/001.err
### S B A T C H --exclude={{ exclude_gpu }}
### S BA TCH --time={{ job_walltime }}
### S  B A TCH --cpus-per-task{{ n_cores }}

date=2022-12-01
LOGGING_DIR=/share/data/willett-group/meliao/pointnet2_logging
QM7_DATA_FP=~/projects/invariant-random-features-code/data/qm7/qm7.mat

source  ~/conda_init.sh
conda activate fourier_neural_operator 
python train_PointNet_QM7.py \
--data_fp $QM7_DATA_FP \
--metadata_record_fp ${LOGGING_DIR}/regression_QM7/${date}_hyperparam_opt.txt \
--log_dir 001 \
--base_log_dir $LOGGING_DIR \
--n_train 5732 \
--n_test 1433 \
--learning_rate 0.1 \
--epoch 2
