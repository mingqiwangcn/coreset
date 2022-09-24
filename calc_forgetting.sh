if [ "$#" -ne 2 ]; then
    echo "Usage: ./calc_forgetting.sh <dataset> <mode>"
    exit
fi
dataset=$1
mode=$2
if [ "${mode}" = "dev" ]; then
    train_file=/home/cc/code/open_table_discovery/table2question/dataset/${dataset}/sql_data/${mode}/rel_graph/fusion_retrieved_tagged_fg.jsonl
else
    train_file=/home/cc/code/open_table_discovery/table2question/dataset/${dataset}/sql_data/${mode}/rel_graph/fusion_retrieved_tagged.jsonl
fi
out_dir=output/forgetting/${dataset}/${mode}
echo ${train_file}
echo ${out_dir}
python ./forgetting_table.py \
    --work_dir ~/code \
    --train_file ${train_file} \
    --out_dir ${out_dir}
