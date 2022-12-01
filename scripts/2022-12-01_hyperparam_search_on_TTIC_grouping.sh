#!/bin/bash
#SBATCH --job-name=pointnet2_grouping
#SBATCH --partition=contrib-gpu
#SBATCH --output=logs/grouping.out
#SBATCH --error=logs/grouping.err
### S B A T C H --exclude={{ exclude_gpu }}
### S BA TCH --time={{ job_walltime }}
### S  B A TCH --cpus-per-task{{ n_cores }}

date=2022-12-01
LOGGING_DIR=/share/data/willett-group/meliao/pointnet2_logging
QM7_DATA_FP=~/projects/invariant-random-features-code/data/qm7/qm7.mat

source  ~/conda_init.sh
conda activate fourier_neural_operator 

# Want to increase the nsample and ncentroid params to get a more expressive model
python train_PointNet_QM7.py \
--data_fp $QM7_DATA_FP \
--metadata_record_fp ${LOGGING_DIR}/regression_QM7/${date}_hyperparam_opt.txt \
--log_dir 004 \
--base_log_dir $LOGGING_DIR \
--n_train 5732 \
--n_test 1433 \
--learning_rate 0.1 \
--epoch 100 \
--n_centroids_1 15 \
--msg_radii_1 2 4 8 \
--msg_nsample_1 8 16 32 \
--n_centroids_2 10 \
--msg_radii_2 2 4 8 \
--msg_nsample_2 4 8 16


# python train_PointNet_QM7.py \
# --data_fp $QM7_DATA_FP \
# --metadata_record_fp ${LOGGING_DIR}/regression_QM7/${date}_hyperparam_opt.txt \
# --log_dir 005 \
# --base_log_dir $LOGGING_DIR \
# --n_train 10 \
# --n_test 1433 \
# --learning_rate 0.01 \
# --epoch 100

# python train_PointNet_QM7.py \
# --data_fp $QM7_DATA_FP \
# --metadata_record_fp ${LOGGING_DIR}/regression_QM7/${date}_hyperparam_opt.txt \
# --log_dir 006 \
# --base_log_dir $LOGGING_DIR \
# --n_train 10 \
# --n_test 1433 \
# --learning_rate 1.0 \
# --epoch 100
