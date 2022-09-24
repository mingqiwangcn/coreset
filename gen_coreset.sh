if [ "$#" -ne 5 ]; then
    echo "Usage: ./gen_coreset.sh <dataset> <mode> <coreset_tag> <coreset_size> <best_step>"
    exit
fi
dataset=$1
mode=$2
coreset_tag=$3
best_step=$5
if [ "${mode}" = "dev" ]; then
    coreset_size=$4    
    data_file=/home/cc/code/open_table_discovery/table2question/dataset/${dataset}/sql_data/${mode}/rel_graph/fusion_retrieved_tagged_fg.jsonl
fi
out_dir=output/forgetting/${dataset}/${mode}
echo ${data_file}
echo ${out_dir}
python ./data_stat.py \
    --data_file ${data_file} \
    --dataset ${dataset} \
    --mode ${mode} \
    --train_name fg_data \
    --coreset_tag ${coreset_tag} \
    --coreset_size ${coreset_size} \
    --best_step ${best_step} 
