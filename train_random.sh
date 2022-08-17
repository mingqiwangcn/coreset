if [ "$#" -ne 2 ]; then
    echo "Usage: ./train.sh <dataset> <tag>"
    exit
fi
dataset=$1
tag=$2
python ../fusion_reader/train_reader.py \
        --train_data ../data/${dataset}/coreset/train_data_p_5_num_${tag}.jsonl \
        --eval_data ../data/${dataset}/coreset/dev_data.jsonl \
        --model_size base \
        --per_gpu_batch_size 1 \
        --n_context 10 \
        --name train_random_p_5_${tag} \
        --checkpoint_dir ./output/forgetting/${dataset}/ \
        --use_auto_steps 1 \
        --epoch_ckp_num 2 \
        --max_epoch 5 \

