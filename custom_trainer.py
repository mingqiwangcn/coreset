import json
import os
from tqdm import tqdm
import glob

class CustomDataset:
    def __init__(self, data):
        self.data = data
    
    def get_example(self, idx):
        return self.data[idx]   
 
def load_data(data_dir):
    file_pattern = os.path.join(data_dir, 'pred_step_*.jsonl')
    data_file_lst = glob.glob(file_pattern)
    step_data_map = {}
    for data_file in tqdm(data_file_lst):
        step = int(data_file.split('_')[-1].split('.')[0])
        step_data = []
        with open(data_file) as f:
            for line in f:
                item = json.loads(line)
                item['id'] = item['qid']
                step_data.append(item)
        step_data_map[step] = step_data 
    return step_data_map 


def main(opts, coreset_method):
    step_data_map = load_data(opts.data_dir)
    step_lst = list(step_data_map.keys())
    step_lst.sort()
    
    first_step_data = step_data_map[step_lst[0]]
    coreset_method.init_data(first_step_data)
   
    bsz = 1 
    for step in tqdm(step_lst):
        step_data = step_data_map[step]
        cus_data = CustomDataset(step_data)
        N = len(step_data)
        for i in range(N):
            idxes = [i]
            coreset_metrics = [step_data[i]['correct']]
            step_info = {
                'step':step,
                'batch':None
            }
            coreset_method.do(cus_data, idxes, coreset_metrics, step_info) 
        
        coreset_method.on_checkpoint(step_info)
