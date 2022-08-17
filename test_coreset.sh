if [ "$#" -ne 3 ]; then
    echo "Usage: ./test_coreset.sh <dataset> <train_name> <tag>"
    exit
fi
dataset=$1
train_name=$2
tag=$3
python ../fusion_reader/test_reader.py \
        --model_path ./output/forgetting/${dataset}/${train_name}_coreset_${tag}/checkpoint/best_dev \
        --eval_data ../data/${dataset}/coreset/test_data.jsonl \
        --per_gpu_batch_size 1 \
        --n_context 10 \
        --name test \
        --checkpoint_dir ./output/forgetting/${dataset}/${train_name} \
