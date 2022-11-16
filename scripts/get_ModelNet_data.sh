DATA_DIR=data

mkdir -p $DATA_DIR
cd $DATA_DIR
wget https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip --no-check-certificate
unzip modelnet40_normal_resampled.zip 
rm modelnet40_normal_resampled.zip