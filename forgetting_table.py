import logging
#logging.basicConfig(level=logging.ERROR)
import json
from tqdm import tqdm
import random
import os
import argparse
import uuid
import finetune_table_retr as model_trainer
import custom_trainer

def read_config():
    with open('../open_table_discovery/trainer.config') as f:
       config = json.load(f)
    return config

def get_train_opt(args):
    work_dir = args.work_dir 
    train_itr = 0
    train_file = args.train_file
    eval_file = None 
    config = read_config() 
    checkpoint_dir = args.out_dir 
    bnn_opt = 1 #int(config['bnn'])
    if bnn_opt:
        checkpoint_name = 'fg_data_bnn'
    else:
        checkpoint_name = 'fg_data'
     
    train_args = argparse.Namespace(sql_batch_no=train_itr,
                                    do_train=True,
                                    model_path=os.path.join(work_dir, 'models/tqa_reader_base'),
                                    train_data=train_file,
                                    eval_data=eval_file,
                                    n_context=int(config['retr_top_n']),
                                    per_gpu_batch_size=int(config['train_batch_size']),
                                    per_gpu_eval_batch_size=int(config['train_batch_size']),
                                    cuda=0,
                                    name=checkpoint_name,
                                    checkpoint_dir=checkpoint_dir,
                                    max_epoch=1,
                                    patience_steps=int(config['patience_steps']),
                                    ckp_steps=int(config['ckp_steps']),
                                    bnn=bnn_opt,
                                    text_maxlength=int(config['text_maxlength']),
                                    fusion_retr_model=None,
                                    prior_model=None,
                                    )
    return train_args

    
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
        step_data_dir = os.path.join(out_dir, 'step_data')
        self.out_dir = step_data_dir
        self.counter = 0
        self.mean_changes = 0
    
    def set_epoch_steps(self, epoch_steps):
        self.epoch_steps = epoch_steps
        self.uni_waiting_steps = int(self.epoch_steps / 30)
     
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
        
        #if item[CoresetMethod.Not_Change_Steps_Key] >= 2:
        #self.coreset_2_other(item, step)
    
    def get_coreset(self, train_qid_batch):
        if train_qid_batch is not None:
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
        #waiting_steps_1 = min(1, self.epoch_steps) #min(1, self.epoch_steps)
        #waiting_steps_2 = min(10, self.epoch_steps)
        item[CoresetMethod.Reschedule_Step_Key] = step + self.uni_waiting_steps
        #item[CoresetMethod.Reschedule_Step_Key] = step +  random.randint(waiting_steps_1, waiting_steps_2) 
        
    def other_2_coreset(self, qid):
        del self.other_queue[qid]
        self.coreset_queue[qid] = 1
        self.data_stat[qid][CoresetMethod.Not_Change_Steps_Key] = 0 
    
    def init_data(self, data):
        self.coreset_queue = {}
        self.other_queue = {}
        if not os.path.isdir(self.out_dir):
            os.makedirs(self.out_dir)
        for idx, item in enumerate(data):
            qid = item['id']
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
        self.counter += 1
        for offset, idx in enumerate(idxes):
            qid = dataset.get_example(idx)['id']
            acc = coreset_metrics[offset] 
            self.update_forgettings(qid, acc, step_info)

    def on_checkpoint(self, step_info):
        step = step_info['step']
        #self.reschedule(step) 
        self.write_stat(step_info)
         
    def write_stat(self, step_info):
        step = step_info['step']
        file_name = 'forgetting_step_%d.jsonl' % step
        out_stat_file = os.path.join(self.out_dir, file_name)
        with open(out_stat_file, 'w') as f_o:
            for qid in self.data_stat:
                item_stat = self.data_stat[qid]
                item_stat['train_points'] = step_info['batch']
                f_o.write(json.dumps(item_stat) + '\n')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work_dir', type=str)
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--part_no', type=str)
    parser.add_argument('--out_dir', type=str)
    args = parser.parse_args()
    return args

def main():
    global logger
    logger = logging.getLogger(__name__)
    args = get_args()
    #opt = get_train_opt(args)
    #out_dir = os.path.join(opt.checkpoint_dir, opt.name)
   
    out_dir = 'output/forgetting/%s/train_0/%s/fg_data_bnn' % (args.dataset, args.part_no)
    opts = argparse.Namespace(out_dir=out_dir,
                              data_dir=args.data_dir)
    if os.path.isdir(opts.out_dir):
        print('%s already exists' % out_dir)
        return

    method = CoresetMethod(out_dir)
    custom_trainer.main(opts, coreset_method=method)

if __name__ == '__main__':
    main()

