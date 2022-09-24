if [ "$#" -ne 5 ]; then
    echo "Usage: ./gen_coreset.sh <dataset> <mode> <part> <coreset_size> <best_step>"
    exit
fi
dataset=$1
mode=$2
part=$3
coreset_tag=coreset_fg
coreset_size=$4 
best_step=$5
if [ "${mode}" = "dev" ]; then
    data_file=/home/cc/code/open_table_discovery/table2question/dataset/${dataset}/sql_data/${mode}/rel_graph/fusion_retrieved_tagged_fg.jsonl
else
    data_file=/home/cc/code/open_table_discovery/table2question/dataset/${dataset}/sql_data/${mode}/rel_graph/data_parts/${part}.jsonl
fi
out_dir=output/forgetting/${dataset}/${mode}/${part}
echo ${data_file}
echo ${out_dir}
python ./data_stat.py \
    --data_file ${data_file} \
    --dataset ${dataset} \
    --mode ${mode} \
    --part ${part} \
    --coreset_tag ${coreset_tag} \
    --coreset_size ${coreset_size} \
    --best_step ${best_step} 
