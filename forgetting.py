import logging
#logging.basicConfig(level=logging.ERROR)
import json
from tqdm import tqdm
import random
import os
import argparse
import uuid
import train_reader
from src.options import Options 

def get_train_opt():
    options = Options()
    options.add_reader_options()
    options.add_optim_options()
    opt = options.parse()
    data_dir = 'output/forgetting'
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)
    percentage = 5
    train_name = 'train_%d' % percentage
    train_file = '../data/NQ/coreset/train_data_percent_%d.jsonl' % percentage
    opt.train_data = train_file
    opt.eval_data = '../data/NQ/coreset/dev_data.jsonl'
    opt.model_size= 'base'
    opt.per_gpu_batch_size = 1 
    opt.n_context = 10
    opt.name=train_name
    opt.checkpoint_dir=data_dir
    opt.use_auto_steps = 1
    opt.epoch_ckp_num = 2 
    opt.max_epoch = 5
    return opt

data_stat = {}
prev_acc_key = 'prev_acc'
forgetting_key = 'forgetting'
update_cnt_key = 'update_cnt'

def update_forgettings(qid, acc, step):
    if qid not in data_stat:
        data_stat[qid] = {
            'qid':qid,
            prev_acc_key:0,
            forgetting_key:0,
            update_cnt_key:0,
            'first_correct_step':None
        }
    item = data_stat[qid]
    if item[prev_acc_key] > acc:
        item[forgetting_key] += 1
    item[prev_acc_key] = acc
    item[update_cnt_key] += 1
    if acc > 0:
        if item['first_correct_step'] is None:
            item['first_correct_step'] = step

class CoresetMethod:
    def __init__(self, out_dir):
        self.call_back = train_reader.evaluate_train
        self.out_dir = out_dir

    def do(self, dataset, idxes, coreset_metrics, step):
        for offset, idx in enumerate(idxes):
            qid = dataset.get_example(idx)['qid']
            acc = coreset_metrics[offset] 
            update_forgettings(qid, acc, step)

    def on_checkpoint(self, step):
        write_stat(self.out_dir, step)
         
def main():
    opt = get_train_opt()
    out_dir = os.path.join(opt.checkpoint_dir, opt.name)
    if os.path.isdir(out_dir):
        print('%s already exists' % out_dir)
        return
    method = CoresetMethod(out_dir)
    train_reader.main(opt, coreset_method=method)
    write_stat(out_dir)

def write_stat(out_dir, step=None):
    file_name = 'forgetting.jsonl'
    if step is not None:
        file_name = 'forgetting_step_%d.jsonl' % step
    out_stat_file = os.path.join(out_dir, file_name)
    with open(out_stat_file, 'w') as f_o:
        for qid in data_stat:
            item_stat = data_stat[qid]
            f_o.write(json.dumps(item_stat) + '\n')

if __name__ == '__main__':
    main()

