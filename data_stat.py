import json
import glob
import csv
import numpy as np
from tqdm import tqdm
import os

def get_steps(train_name, upto_step):
    data_file = './output/forgetting/%s/forgetting_step_*.jsonl' % train_name
    file_lst = glob.glob(data_file)
    
    step_lst = []
    for file_info in file_lst:
        file_base_name = os.path.basename(file_info)
        pos_1 = file_base_name.index('_step_') + len('_step_')
        pos_2 = file_base_name.rindex('.jsonl')
        step = int(file_base_name[pos_1:pos_2])
        if step <= upto_step:
            step_lst.append(step)
    step_lst.sort()
    return step_lst

def get_step_forgettings(train_name, step):
    data_map = {}
    data_file = './output/forgetting/%s/forgetting_step_%d.jsonl' % (train_name, step)
    with open(data_file) as f:
        for line in f:
            item = json.loads(line)
            qid = item['qid']
            data_map[qid] = item
    return data_map 

def verify_serials(upto_step):
    forgettings = get_step_forgettings(upto_step) 
    serial_map = get_forgetting_serials(upto_step) 
    serial_sum_data = {}
    for step in serial_map:
        for qid in serial_map[step]:
            if qid not in serial_sum_data:
                serial_sum_data[qid] = 0
            serial_sum_data[qid] += 1
    
    for qid in forgettings:
        item = forgettings[qid]
        if item['forgetting'] > 0:
            #import pdb; pdb.set_trace()
            assert(serial_sum_data[qid] == item['forgetting'])
    
    print('verify_serials ok')

def get_forgetting_serials(upto_step):
    serial_map = {}
    step_lst = get_steps(upto_step)
    for step in tqdm(step_lst, desc='serial data'):
        if step < 1:
            continue
        serial_map[step] = []
        pre_step_forgettings = get_step_forgettings(step - 1)
        cur_step_forgettings = get_step_forgettings(step) 
        for qid in cur_step_forgettings:
            pre_item = pre_step_forgettings[qid]
            cur_item = cur_step_forgettings[qid]
            if cur_item['prev_acc'] < pre_item['prev_acc']:
                serial_map[step].append(qid)
   
    return serial_map 

def sum_block_forgettings(serial_map, step_blocks):
    forgettings = 0
    for step in step_blocks:
        forgettings += len(serial_map[step]['forgetting_points'])
    return forgettings 

def write_serial_forgettings(upto_step):
    serial_map = get_forgetting_serials(upto_step)
    step_sorted = sorted(list(serial_map.keys()))
    with open('forgetting_serials.jsonl', 'w') as f_o:
        for step in step_sorted:
            out_item = {
                'step':step,
                'forgetting_points':serial_map[step]
            }
            f_o.write(json.dumps(out_item) + '\n') 

def read_serial_forgettings():
    data_map = {}
    data_file = 'forgetting_serials.jsonl'
    with open(data_file) as f:
        for line in f:
            item = json.loads(line)
            step = item['step']
            data_map[step] = item
    return data_map

def gen_serial_report():
    serial_data = read_serial_forgettings()
    col_names = ['step', 'forgetting']
    step_sorted = sorted(list(serial_data.keys()))
    with open('forgetting_serials_step_detail.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(col_names)
        for step in step_sorted:
            forgettings = len(serial_data[step]['forgetting_points'])
            writer.writerow([str(step), forgettings])

def gen_point_step_forgettings():
    serial_data = read_serial_forgettings()
    point_step_map = {}
    for step in serial_data:
        data_points = serial_data[step]['forgetting_points']
        for qid in data_points:
            if qid not in point_step_map:
                point_step_map[qid] = []
            point_step_map[qid].append(step)
    
    forgetting_point_map = {}
    for qid in point_step_map:
        forgetting_cnt = len(point_step_map[qid])
        if forgetting_cnt not in forgetting_point_map:
            forgetting_point_map[forgetting_cnt] = []
        forgetting_point_map[forgetting_cnt].append(qid)
   
    forgetting_lst = sorted(list(forgetting_point_map.keys()))
    for forgetting in forgetting_lst:
        out_file = './point_forgettings/point_forgettings_%d.jsonl' % forgetting
        with open(out_file, 'w') as f_o:
            points = forgetting_point_map[forgetting]
            for qid in points:
                steps = point_step_map[qid]
                for step in steps:
                    out_item = {
                        'point':qid,
                        'step':step
                    }
                    f_o.write(json.dumps(out_item) + '\n')

def gen_serial_block_report(block_size):
    serial_data = read_serial_forgettings()
    step_sorted = sorted(list(serial_data.keys()))
    col_names = ['step', 'forgetting']
    with open('forgetting_serials_block_%d.csv' % block_size, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(col_names)
        for offset in range(0, len(step_sorted), block_size):
            step_block_desc = '%d-%d' % (offset, offset + block_size - 1)
            step_blocks = step_sorted[offset:(offset+block_size)]
            forgettings = sum_block_forgettings(serial_data, step_blocks)
            mean_forgettings = forgettings / len(step_blocks)
            writer.writerow([step_block_desc, mean_forgettings])

def get_forgetting_dist(train_name):
    data_file = './output/forgetting/%s/forgetting_sorted.jsonl' % train_name
    col_names = ['forgetting', 'count']
    stat_map = {}
    with open(data_file) as f:
        for line in f:
            item = json.loads(line)
            fogetting = item['forgetting']
            if fogetting == 0:
                if item['first_correct_step'] is not None:
                    key = '0-learnable'
                else:
                    key = '0-unlearnable'
            else:
                key = '%d' % fogetting
            if key not in stat_map:
                stat_map[key] = 0
            stat_map[key] += 1
    
    with open('forgetting_stat.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(col_names)
        for key in stat_map:
            writer.writerow([key, stat_map[key]])
     

def read_data(dataset, base_name):
    data_file = '../data/%s/coreset/train_data_%s.jsonl' % (dataset, base_name) # percent_5
    data_map = {}
    with open(forgetting_file) as f:
        for line in f:
            item = json.loads(line)
            qid = item['qid']
            data_map[qid] = item
    return data_map


def gen_forgetting_data(dataset, train_name, step):
    update_cnt_lst = [] 
    forgetting_lst = []
    data_learnable = []
    data_unlearnable = []
    data_file = './output/forgetting/%s/%s/step_data/forgetting_step_%d.jsonl' % (dataset, train_name, step)
    with open(data_file) as f:
        for line in f:
            item = json.loads(line)
            update_cnt_lst.append(item['update_cnt'])
            if item['forgetting'] > 0:
                forgetting_lst.append(item['forgetting'])
            if item['first_correct_step'] is not None:
                data_learnable.append(item)
            else:
                data_unlearnable.append(item)

    sort_key = lambda a: (a['forgetting'], a['first_correct_step'])
    data_learnable_sorted = sorted(data_learnable, key=sort_key)
    
    out_data = data_learnable_sorted + data_unlearnable
    with open('./output/forgetting/%s/%s/forgetting_sorted.jsonl' % (dataset, train_name), 'w') as f_o:
        for item in out_data:
            f_o.write(json.dumps(item) + '\n')
    
    update_cnt_min = np.min(update_cnt_lst)
    update_cnt_max = np.max(update_cnt_lst)
    update_cnt_mean = np.mean(update_cnt_lst)
    update_cnt_std = np.std(update_cnt_lst)
    update_cnt_median = np.median(update_cnt_lst)

    print('update_cnt_min=%d, update_cnt_max=%d, update_cnt_mean=%d, update_cnt_std=%d, update_cnt_median=%d' % (
           update_cnt_min, update_cnt_max, update_cnt_mean, update_cnt_std, update_cnt_median))

    print('forgetting_min=%d, forgetting_max=%d, forgetting_mean=%d, forgetting_std=%d, forgetting_median=%d' % (
           np.min(forgetting_lst),
           np.max(forgetting_lst),
           np.mean(forgetting_lst),
           np.std(forgetting_lst),
           np.median(forgetting_lst))
    )


def gen_coreset(dataset, base_name, train_name, coreset_tag, up_to_rows, strategy_func):
    data_file = '../data/%s/coreset/train_data_%s.jsonl' % (dataset, base_name) # percent_5
    data = []
    forgetting_file = './output/forgetting/%s/%s/forgetting_sorted.jsonl' % (dataset, train_name)
    with open(forgetting_file) as f:
        for line in f:
            item = json.loads(line)
            data.append(item)
    
    out_qid_set = strategy_func(data)
    out_dir = './output/forgetting/%s/%s/coreset/' % (dataset, train_name)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    file_name = '%s_coreset_%s.jsonl' % (train_name, coreset_tag)
    out_file = os.path.join(out_dir, file_name)
    f_o = open(out_file, 'w')
    with open(data_file) as f_2:
        for line in f_2:
            item = json.loads(line)
            if item['qid'] in out_qid_set:
                f_o.write(line)
    f_o.close()


def remove_zero_forgetting(data):
    qid_set = set()
    for item in data:
        if item['forgetting'] > 0:
            qid_set.add(item['qid'])
    return qid_set


def use_unlearnable_only(data):
    qid_set = set()
    for item in data:
        if item['first_correct_step'] is None:
            qid_set.add(item['qid'])
    return qid_set


def use_learnable_only(data):
    qid_set = set()
    for item in data:
        if item['first_correct_step'] is not None:
            qid_set.add(item['qid'])
    return qid_set


def main():
    dataset = 'NQ'
    base_name = 'percent_5'
    train_name = 'train_5'
    best_steps = 5937
    gen_forgetting_data(dataset, train_name, best_steps)
    gen_coreset(dataset, base_name, train_name, 'forgettable', None, remove_zero_forgetting)
    gen_coreset(dataset, base_name, train_name, 'never_learnt', None, use_unlearnable_only)
    gen_coreset(dataset, base_name, train_name, 'forgettable_unforgettable', None, use_learnable_only)
    #get_forgetting_dist(11874) 
    #verify_serials(11874)
    #write_serial_forgettings(11874)
    #gen_serial_report()
    #gen_serial_block_report(200)
    #gen_point_step_forgettings()

if __name__ == '__main__':
    main()
