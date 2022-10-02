if [ "$#" -ne 4 ]; then
    echo "Usage: ./gen_coreset.sh <dataset> <mode> <part> <best_step>"
    exit
fi
dataset=$1
mode=$2
part=$3
coreset_tag=coreset_learnable
best_step=$4
data_file=/home/cc/code/open_table_discovery/table2question/dataset/${dataset}/sql_data/${mode}/rel_graph/data_parts/${part}.jsonl
out_dir=output/forgetting/${dataset}/${mode}/${part}
echo ${data_file}
echo ${out_dir}
python ./data_stat.py \
    --data_file ${data_file} \
    --dataset ${dataset} \
    --mode ${mode} \
    --part ${part} \
    --coreset_tag ${coreset_tag} \
    --best_step ${best_step} 
