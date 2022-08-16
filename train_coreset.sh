if [ "$#" -ne 3 ]; then
    echo "Usage: ./train.sh <dataset> <train_name> <tag>"
    exit
fi
dataset=$1
train_name=$2
tag=$3
python ../fusion_reader/train_reader.py \
        --train_data ./output/forgetting/${dataset}/${train_name}/coreset/${train_name}_coreset_${tag}.jsonl \
        --eval_data ../data/${dataset}/coreset/dev_data.jsonl \
        --model_size base \
        --per_gpu_batch_size 1 \
        --n_context 10 \
        --name train_coreset_${tag} \
        --checkpoint_dir ./output/forgetting/${dataset} \
        --use_auto_steps 1 \
        --epoch_ckp_num 2 \
        --max_epoch 5 \

