if [ "$#" -ne 2 ]; then
    echo "Usage: ./test_random.sh <dataset> <train_name>"
    exit
fi
dataset=$1
train_name=$2
python ../fusion_reader/test_reader.py \
        --model_path ./output/forgetting/${dataset}/${train_name}/checkpoint/best_dev \
        --eval_data ../data/${dataset}/coreset/test_data.jsonl \
        --per_gpu_batch_size 1 \
        --n_context 10 \
        --name test \
        --checkpoint_dir ./output/forgetting/${dataset}/${train_name} \
