date=2022-12-02

export OMP_NUM_THREADS=2

python train_PointNet_QM7.py \
--data_fp /local/meliao/projects/invariant-random-features/data/qm7/qm7.mat \
--metadata_record_fp log/tmp/2022-12-02_test.txt \
--base_log_dir log/tmp/ \
--n_train 500 \
--n_test 100 \
--epoch 10 \
--save_weights_every_epoch