date=2022-11-30

export OMP_NUM_THREADS=24

python train_PointNet_QM7.py \
--data_fp ~/projects/invariant-random-features-code/data/qm7/qm7.mat \
--metadata_record_fp log/regression_QM7/2022-11-30_hyperparam_opt.txt \
--log_dir 001 \
--n_train 100 \
--n_test 100 \
--learning_rate 0.1 \
--epoch 10