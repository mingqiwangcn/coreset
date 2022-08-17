if [ "$#" -ne 1 ]; then
    echo "Usage: ./train.sh <tag>"
    exit
fi

tag=$1
python ../fusion_reader/train_reader.py \
        --train_data ../data/NQ/coreset/train_data_p_5_num_${tag}.jsonl \
        --eval_data ../data/NQ/coreset/dev_data.jsonl \
        --model_size base \
        --per_gpu_batch_size 1 \
        --n_context 10 \
        --name train_random_p_5_${tag} \
        --checkpoint_dir ./output \
        --use_auto_steps 1 \
        --epoch_ckp_num 2 \
        --max_epoch 5 \

