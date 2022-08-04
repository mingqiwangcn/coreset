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
    opt.per_gpu_eval_batch_size = 60
    return opt

prev_acc_key = 'prev_acc'
forgetting_key = 'forgetting'
update_cnt_key = 'update_cnt'
first_correct_step_key = 'first_correct_step'
not_change_steps_key = 'not_change_steps'
reschedule_step_key = 'reschedule_step'

class CoresetMethod:
    def __init__(self, out_dir):
        self.data_stat = {}
        self.call_back = train_reader.evaluate_train
        self.out_dir = out_dir
            
    def update_forgettings(self, qid, acc, step_info):
        step = step_info['step']
        item = self.data_stat[qid]
        if item[prev_acc_key] is not None:
            if item[prev_acc_key] > acc:
                item[forgetting_key] += 1
                item[not_change_steps_key] = 0 
            elif item[prev_acc_key] == acc:
                item[not_change_steps_key] += 1
            else:
                item[not_change_steps_key] = 0

        item[prev_acc_key] = acc
        item[update_cnt_key] += 1
        if acc > 0:
            if item[first_correct_step_key] is None:
                item[first_correct_step_key] = step
        
        if item[not_change_steps_key] >= 2:
            self.coreset_2_other(item, step)
    
    def get_coreset(self):
        qid_lst = [a for a in self.coreset_queue]
        return qid_lst
     
    def coreset_2_other(self, item, step):
        qid = item['qid']
        del self.coreset_queue[qid] 
        self.other_queue[qid] = 1
        item[reschedule_step_key] = step + 16 
        
    def other_2_coreset(self, qid):
        del self.other_queue[qid]
        self.coreset_queue[qid] = 1 
    
    def init_data(self, data):
        self.coreset_queue = {}
        self.other_queue = {}
        for idx, item in enumerate(data):
            qid = item['qid']
            self.data_stat[qid] = {
                'qid':qid,
                prev_acc_key:None,
                forgetting_key:0,
                update_cnt_key:0,
                first_correct_step_key:None,
                not_change_steps_key:0,
                reschedule_step_key:None
            }
            self.coreset_queue[qid] = 1
    
    def reschedule(self, step):
        qid_lst = [a for a in self.other_queue]
        for qid in qid_lst:
            if step >= self.data_stat[qid][reschedule_step_key]:
                self.other_2_coreset(qid)

    def do(self, dataset, idxes, coreset_metrics, step_info):
        for offset, idx in enumerate(idxes):
            qid = dataset.get_example(idx)['qid']
            acc = coreset_metrics[offset] 
            self.update_forgettings(qid, acc, step_info)

    def on_checkpoint(self, step_info):
        step = step_info['step']
        self.reschedule(step) 
        self.write_stat(step)
         
    def write_stat(self, step):
        file_name = 'forgetting_step_%d.jsonl' % step
        out_stat_file = os.path.join(self.out_dir, file_name)
        with open(out_stat_file, 'w') as f_o:
            for qid in self.data_stat:
                item_stat = self.data_stat[qid]
                f_o.write(json.dumps(item_stat) + '\n')
def main():
    opt = get_train_opt()
    out_dir = os.path.join(opt.checkpoint_dir, opt.name)
    if os.path.isdir(out_dir):
        print('%s already exists' % out_dir)
        return
    method = CoresetMethod(out_dir)
    train_reader.main(opt, coreset_method=method)


if __name__ == '__main__':
    main()

