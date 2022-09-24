if [ "$#" -ne 3 ]; then
    echo "Usage: ./calc_forgetting.sh <dataset> <mode> <part_no>"
    exit
fi
dataset=$1
mode=$2
if [ "${mode}" = "dev" ]; then
    data_file=/home/cc/code/open_table_discovery/table2question/dataset/${dataset}/sql_data/${mode}/rel_graph/fusion_retrieved_tagged_fg.jsonl
else
    part_no=$3
    data_file=/home/cc/code/open_table_discovery/table2question/dataset/${dataset}/sql_data/${mode}/rel_graph/data_parts/${part_no}.jsonl
fi
out_dir=output/forgetting/${dataset}/${mode}/${part_no}
echo ${data_file}
echo ${out_dir}
python ./forgetting_table.py \
    --work_dir ~/code \
    --train_file ${data_file} \
    --out_dir ${out_dir}
