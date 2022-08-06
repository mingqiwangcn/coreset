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


class CoresetMethod:
    Prev_Acc_Key = 'prev_acc'
    Forgetting_Key = 'forgetting'
    Update_Cnt_Key = 'update_cnt'
    First_Correct_Step_Key = 'first_correct_step'
    Not_Change_Steps_Key = 'not_change_steps'
    Reschedule_Step_Key = 'reschedule_step'
    Flip_Cnt_Key = 'flip_cnt'
    
    def __init__(self, out_dir):
        self.data_stat = {}
        self.call_back = train_reader.evaluate_train
        self.out_dir = out_dir
            
    def update_forgettings(self, qid, acc, step_info):
        step = step_info['step']
        item = self.data_stat[qid]
        if item[CoresetMethod.Prev_Acc_Key] is not None:
            if item[CoresetMethod.Prev_Acc_Key] > acc:
                item[CoresetMethod.Forgetting_Key] += 1
                item[CoresetMethod.Not_Change_Steps_Key] = 0
                item[CoresetMethod.Flip_Cnt_Key] += 1 
            elif item[CoresetMethod.Prev_Acc_Key] == acc:
                item[CoresetMethod.Not_Change_Steps_Key] += 1
            else:
                item[CoresetMethod.Not_Change_Steps_Key] = 0
                item[CoresetMethod.Flip_Cnt_Key] += 1

        item[CoresetMethod.Prev_Acc_Key] = acc
        item[CoresetMethod.Update_Cnt_Key] += 1
        if acc > 0:
            if item[CoresetMethod.First_Correct_Step_Key] is None:
                item[CoresetMethod.First_Correct_Step_Key] = step
        
        if item[CoresetMethod.Not_Change_Steps_Key] >= 2:
            self.coreset_2_other(item, step)
    
    def get_coreset(self, train_qid_batch):
        for qid in train_qid_batch:
            if qid not in self.coreset_queue:
                self.other_2_coreset(qid)
        qid_lst = [a for a in self.coreset_queue]
        assert (len(qid_lst) > 0)
        return qid_lst
     
    def coreset_2_other(self, item, step):
        qid = item['qid']
        del self.coreset_queue[qid] 
        self.other_queue[qid] = 1
        item[CoresetMethod.Reschedule_Step_Key] = step + 32 
        
    def other_2_coreset(self, qid):
        del self.other_queue[qid]
        self.coreset_queue[qid] = 1
        self.data_stat[qid][CoresetMethod.Not_Change_Steps_Key] = 0 
    
    def init_data(self, data):
        self.coreset_queue = {}
        self.other_queue = {}
        for idx, item in enumerate(data):
            qid = item['qid']
            self.data_stat[qid] = {
                'qid':qid,
                CoresetMethod.Prev_Acc_Key:None,
                CoresetMethod.Forgetting_Key:0,
                CoresetMethod.Update_Cnt_Key:0,
                CoresetMethod.First_Correct_Step_Key:None,
                CoresetMethod.Not_Change_Steps_Key:0,
                CoresetMethod.Reschedule_Step_Key:None,
                CoresetMethod.Flip_Cnt_Key:0
            }
            self.coreset_queue[qid] = 1
    
    def reschedule(self, step):
        qid_lst = [a for a in self.other_queue]
        for qid in qid_lst:
            if step >= self.data_stat[qid][CoresetMethod.Reschedule_Step_Key]:
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

