# absolute path that contains all datasets
# DATA_ROOT=/mnt/sda
DATA_ROOT=datasets
#
## nyu
#DATASET=$DATA_ROOT/sc_depth_dataset
DATASET=$DATA_ROOT/replica_v1
#ATASET=datasets/nyu
CONFIG=configs/v2/nyu.txt

# training options
EXPNAME=habitat_replica_b8

python train.py --config $CONFIG --dataset_dir $DATASET --exp_name $EXPNAME
