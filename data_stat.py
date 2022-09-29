import json
import glob
import csv
import numpy as np
from tqdm import tqdm
import os
import argparse

def get_steps(dataset, train_name, upto_step):
    data_file = './output/forgetting/%s/%s/step_data/forgetting_step_*.jsonl' % (dataset, train_name)
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

def get_step_forgettings(dataset, mode, train_name, part, step):
    data_map = {}
    data_file = './output/forgetting/%s/%s/%s/%s/step_data/forgetting_step_%d.jsonl' % (dataset, mode, part, train_name, step)
    with open(data_file) as f:
        for line in f:
            item = json.loads(line)
            qid = item['qid']
            data_map[qid] = item
    return data_map 

def verify_serials(dataset, train_name, upto_step):
    print('begining verify_serials')
    forgettings = get_step_forgettings(dataset, train_name, upto_step)
    serial_map = get_forgetting_serials(dataset, train_name, upto_step) 
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
    
    print('verify_serials %s %s ok' % (dataset, train_name))

def get_forgetting_serials(dataset, train_name, upto_step):
    serial_map = {}
    step_lst = get_steps(dataset, train_name, upto_step)
    for step in tqdm(step_lst, desc='serial data'):
        if step < 1:
            continue
        serial_map[step] = []
        pre_step_forgettings = get_step_forgettings(dataset, train_name, step - 1)
        cur_step_forgettings = get_step_forgettings(dataset, train_name, step) 
        for qid in cur_step_forgettings:
            pre_item = pre_step_forgettings[qid]
            cur_item = cur_step_forgettings[qid]
            if cur_item['prev_acc'] < pre_item['prev_acc']:
                serial_map[step].append(qid)
   
    return serial_map 


def write_serial_forgettings(dataset, train_name, upto_step):
    serial_map = get_forgetting_serials(dataset, train_name, upto_step)
    step_sorted = sorted(list(serial_map.keys()))
    out_dir = './output/forgetting/%s/%s/report' % (dataset, train_name)
    out_file = os.path.join(out_dir, 'forgetting_serials.jsonl')
    with open(out_file, 'w') as f_o:
        for step in step_sorted:
            out_item = {
                'step':step,
                'forgetting_points':serial_map[step]
            }
            f_o.write(json.dumps(out_item) + '\n') 

def read_serial_forgettings(dataset, train_name):
    data_dir = './output/forgetting/%s/%s/report' % (dataset, train_name)
    data_file = os.path.join(data_dir, 'forgetting_serials.jsonl')
    data_map = {}
    with open(data_file) as f:
        for line in f:
            item = json.loads(line)
            step = item['step']
            data_map[step] = item
    return data_map

def report_step_forgettings(dataset, train_name):
    serial_data = read_serial_forgettings(dataset, train_name)
    col_names = ['step', 'forgetting']
    step_sorted = sorted(list(serial_data.keys()))
    
    out_dir = './output/forgetting/%s/%s/report' % (dataset, train_name)
    out_file = os.path.join(out_dir, 'forgetting_serials_step_detail.csv')
    with open(out_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(col_names)
        for step in step_sorted:
            forgettings = len(serial_data[step]['forgetting_points'])
            writer.writerow([str(step), forgettings])

def gen_point_step_forgettings(dataset, train_name):
    serial_data = read_serial_forgettings(dataset, train_name)
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
    out_dir = './output/forgetting/%s/%s/report/point_forgettings' % (dataset, train_name)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    for forgetting in forgetting_lst:
        file_name = 'point_forgettings_%d.jsonl' % forgetting
        out_file = os.path.join(out_dir, file_name)
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


def get_forgetting_dist(train_name):
    data_file = './output/forgetting/%s/report/forgetting_sorted.jsonl' % train_name
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


def gen_forgetting_data(dataset, mode, part, step):
    update_cnt_lst = [] 
    forgetting_lst = []
    data_learnable = []
    data_unlearnable = []

    if mode == 'dev':
        data_file = './output/forgetting/%s/%s/fg_data_bnn/step_data/forgetting_step_%d.jsonl' % (dataset, mode, step)
    else:
        data_file = './output/forgetting/%s/%s/%s/fg_data_bnn/step_data/forgetting_step_%d.jsonl' % (dataset, mode, part, step)
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
    out_dir = './output/forgetting/%s/%s/%s/fg_data/report' % (dataset, mode, part)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    file_name = 'forgetting_sorted.jsonl'
    out_file = os.path.join(out_dir, file_name)
    with open(out_file, 'w') as f_o:
        for item in out_data:
            f_o.write(json.dumps(item) + '\n')
   
    print('out report, %s' % out_file) 
    update_cnt_min = np.min(update_cnt_lst)
    update_cnt_max = np.max(update_cnt_lst)
    update_cnt_mean = np.mean(update_cnt_lst)
    update_cnt_std = np.std(update_cnt_lst)
    update_cnt_median = np.median(update_cnt_lst)

    #print('update_cnt_min=%d, update_cnt_max=%d, update_cnt_mean=%d, update_cnt_std=%d, update_cnt_median=%d' % (
    #       update_cnt_min, update_cnt_max, update_cnt_mean, update_cnt_std, update_cnt_median))

    #print('forgetting_min=%d, forgetting_max=%d, forgetting_mean=%d, forgetting_std=%d, forgetting_median=%d' % (
    #       np.min(forgetting_lst),
    #       np.max(forgetting_lst),
    #       np.mean(forgetting_lst),
    #       np.std(forgetting_lst),
    #       np.median(forgetting_lst))
    #)


def gen_coreset(data_file, dataset, mode, part, coreset_tag, coreset_size, strategy_func):
    data = []
    forgetting_file = './output/forgetting/%s/%s/%s/fg_data/report/forgetting_sorted.jsonl' % (dataset, mode, part)
    with open(forgetting_file) as f:
        for line in f:
            item = json.loads(line)
            data.append(item)
    
    out_qid_set = strategy_func(mode, data, coreset_size)
    
    out_exp_dir = '/home/cc/code/open_table_discovery/table2question/dataset/fetaqa/sql_data/%s/rel_graph' % mode
    if mode == 'dev':
        file_name = '%s.jsonl' % (coreset_tag)
    else:
        file_name = 'data_parts/%s_%s.jsonl' % (part, coreset_tag)
    out_file = os.path.join(out_exp_dir, file_name)
    if os.path.isfile(out_file):
        raise ValueError('%s already exists' % out_file)
    f_o = open(out_file, 'w')
    with open(data_file) as f_2:
        for line in f_2:
            item = json.loads(line)
            if item['id'] in out_qid_set:
                f_o.write(line)

    print('out coreset, %s' % out_file)
    f_o.close()


def coreset_fg(mode, data, str_coreset_size_or_ratio):
    if str_coreset_size_or_ratio == 'none':
        coreset_size = None
    else:
        coreset_size_or_ratio = float(str_coreset_size_or_ratio)
        assert(coreset_size_or_ratio > 0)
        if coreset_size_or_ratio < 1:
            coreset_size = int(len(data) * coreset_size_or_ratio)
        else:
            coreset_size = coreset_size_or_ratio
    qid_set = set()
    forgetting_lst = []
    never_learnt_lst = []
    for item in data:
        if item['forgetting'] > 0:
            forgetting_lst.append(item['qid']) 
        elif item['first_correct_step'] is None:
            never_learnt_lst.append(item['qid'])
    if (mode != 'dev') or (coreset_size is None):
        qid_set = set(forgetting_lst + never_learnt_lst)
    else:
        num_forgetting = len(forgetting_lst)
        num_never_learnt = len(never_learnt_lst)
        total = num_forgetting + num_never_learnt 
        if total > coreset_size :
            forgetting_coreset_size = int((num_forgetting / total) * coreset_size)
            never_learn_coreset_size = coreset_size - forgetting_coreset_size
            coreset_qid_lst = forgetting_lst[-forgetting_coreset_size:] + never_learnt_lst[-never_learn_coreset_size:]
            qid_set = set(coreset_qid_lst)
        else:
            qid_set = [a['qid'] for a in data][-coreset_size:]
         
    return qid_set

def remove_unforgettable(data):
    qid_set = set()
    for item in data:
        if (item['forgetting'] > 0) or (item['first_correct_step'] is None):
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
    args = get_args()
    gen_forgetting_data(args.dataset, args.mode, args.part, args.best_step)
    gen_coreset(args.data_file, args.dataset, args.mode, 
                args.part, args.coreset_tag, args.coreset_size, coreset_fg)
    #gen_coreset(dataset, base_name, train_name, 'never_learnt', None, use_unlearnable_only)
    #gen_coreset(dataset, base_name, train_name, 'forgettable_unforgettable', None, use_learnable_only)
    
    #verify_serials(dataset, train_name, best_steps)
    #write_serial_forgettings(dataset, train_name, best_steps)
    #report_step_forgettings(dataset, train_name)
    #gen_point_step_forgettings(dataset, train_name)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--mode', type=str, required=True)
    parser.add_argument('--part', type=str, required=True)
    parser.add_argument('--coreset_tag', type=str, required=True)
    parser.add_argument('--coreset_size', type=str, required=True)
    parser.add_argument('--best_step', type=int, required=True)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
