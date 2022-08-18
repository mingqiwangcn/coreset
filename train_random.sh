if [ "$#" -ne 3 ]; then
    echo "Usage: ./train.sh <dataset> <data_file> <train_name>"
    exit
fi
dataset=$1
data_file=$2
train_name=$3
python ../fusion_reader/train_reader.py \
        --train_data ../data/${dataset}/coreset/${data_file} \
        --eval_data ../data/${dataset}/coreset/dev_data.jsonl \
        --model_size base \
        --per_gpu_batch_size 1 \
        --n_context 10 \
        --name ${train_name} \
        --checkpoint_dir ./output/forgetting/${dataset}/ \
        --use_auto_steps 1 \
        --epoch_ckp_num 2 \
        --max_epoch 20 \

