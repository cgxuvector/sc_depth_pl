# absolute path that contains all datasets
DATA_ROOT=/mnt/sda
#
## nyu
DATASET=$DATA_ROOT/sc_depth_dataset
#ATASET=datasets/nyu
CONFIG=configs/v2/nyu.txt

python train.py --config $CONFIG --dataset_dir $DATASET