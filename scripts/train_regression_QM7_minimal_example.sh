date=2022-11-16

export OMP_NUM_THREADS=2

python train_regression_QM7.py \
--data_fp ~/projects/invariant-random-features-code/data/qm7/qm7.mat \
--n_train 500 \
--n_test 100 