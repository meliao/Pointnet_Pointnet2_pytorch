date=2022-11-23

export OMP_NUM_THREADS=24

python train_regression_QM7.py \
--data_fp ~/projects/invariant-random-features-code/data/qm7/qm7.mat \
--n_train 5732 \
--n_test 1433 \
--epoch 1000