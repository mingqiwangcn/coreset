import logging
logging.basicConfig(level=logging.ERROR)
import json
from tqdm import tqdm
import random
from scipy.stats import bernoulli
import os
import argparse
import uuid
import train_reader
from src.options import Options 

def read_data(data_file):
    data = []
    with open(data_file) as f:
        for line in f:
            item = json.loads(line)
            data.append(item)
    return data

def init_shapley(data):
    shapley_dict = {}
    for item in data:
        qid = item['qid']
        shapley_dict[qid] = {'item':item, 'shapley':None, 'itr':0}
    return shapley_dict

def evaluate(sub_set):
    options = Options()
    options.add_reader_options()
    options.add_optim_options()
    opt = options.parse()
 
    data_dir = 'output/shapley'
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)
    train_name = 'train_%d_%s' % (len(sub_set), str(uuid.uuid4()))
    train_file = os.path.join(data_dir, train_name + '.jsonl')
    with open(train_file, 'w') as f_o:
        for item in sub_set:
            f_o.write(json.dumps(item) + '\n')
   
    opt.train_data = train_file
    opt.eval_data = '../fusion_reader/open_domain_data/NQ/coreset/dev_shapley.jsonl'
    opt.model_size= 'base'
    opt.per_gpu_batch_size = 1 
    opt.n_context = 10
    opt.name=train_name
    opt.checkpoint_dir=data_dir
    opt.use_auto_steps = 1 
    report = train_reader.main(opt)
    accuracy = report['best_dev_em']
    return accuracy

def main():
    out_dir = ''
    data_file = '../fusion_reader/open_domain_data/NQ/coreset/train_data_shapley.jsonl'
    data = read_data(data_file)
    shapley_dict = init_shapley(data)
    T = 100
    base_size_lst = [8, 16, 32, 64, 128]
    for t in tqdm(range(1, T+1)):
        k = random.sample(base_size_lst, 1)[0]
        S_t = random.sample(data, k)
        print('evaluating base set')
        accuracy_base = evaluate(S_t)
        print('%d accuracy base=%.4f' % (t, accuracy_base))
        bsz = 8
        base_qid_set = set([item['qid'] for item in S_t])
        value_data = [a for a in data if a['qid'] not in base_qid_set]
        random.shuffle(value_data) 
        for pos in tqdm(range(0, len(value_data), bsz), desc='compute set value'):
            value_set = data[pos:(pos+bsz)]
            merged_set = S_t + value_set
            print('evaluating merged set')
            accuracy_merged = evaluate(merged_set)
            print('%d accuracy merged=%.4f' % (t, accuracy_merged))
            accuracy_improved = (accuracy_merged - accuracy_base) / len(value_set)
            for value_item in value_set:
                shapley_item = shapley_dict[value_item['qid']]
                itr = shapley_item['itr'] + 1
                shapley_1 = shapley_item['shapley']
                if shapley_1 is None:
                    shapley_1 = 0
                shapley_item['shapley'] = (1/itr) * accuracy_improved + ((itr-1)/itr) * shapley_1
                shapley_item['itr'] = itr 
       
        write_shapley(t, shapley_dict)

def write_shapley(t, shapley_dict):
    out_file = './output/shapley_%d.jsonl' % t
    with open(out_file, 'w') as f_o:
        for qid in tqdm(shapley_dict, desc='write shapley'):
            out_item = {
                'qid':qid,
                'shapley':shapley_dict[qid]['shapley'],
                'itr':shapley_dict[qid]['itr']
            }
            f_o.write(json.dumps(out_item) + '\n')

if __name__ == '__main__':
    main()
