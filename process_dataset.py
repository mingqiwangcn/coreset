import json
from tqdm import tqdm
import os
import random

def get_top_10_passages(mode, data_file, out_file):
    print('loading %s' % data_file)
    with open(data_file) as f:
        data = json.load(f)
    
    if os.path.isfile(out_file):
        raise ValueError('%s already exists' % out_file)

    with open(out_file, 'w') as f_o:
        for row, item in tqdm(enumerate(data), total=len(data)):
            item['qid'] = '%s_%d' % (mode, row)
            top_ctxs = item['ctxs'][:10]
            item['ctxs'] = top_ctxs
            f_o.write(json.dumps(item) + '\n') 

def main_top_10(dataset):
    get_top_10_passages('train', './open_domain_data/%s/train.json' % dataset, 
                        '../data/%s/coreset/train_data.jsonl' % dataset) 
    get_top_10_passages('dev', './open_domain_data/%s/dev.json' % dataset, 
                        '../data/%s/coreset/dev_data.jsonl' % dataset) 
    get_top_10_passages('test', './open_domain_data/%s/test.json' % dataset,
                        '../data/%s/coreset/test_data.jsonl' % dataset) 

def read_data(data_file):
    data = []
    with open(data_file) as f:
        for line in tqdm(f):
            data.append(line)
    return data            

def get_train_percent(dataset):
    data_file = '../data/%s/coreset/train_data.jsonl' % dataset
    train_data = read_data(data_file)
    percent_lst = [5]
    for percent in tqdm(percent_lst):
        out_file = '../data/%s/coreset/train_data_percent_%d.jsonl' % (dataset, percent)
        if os.path.isfile(out_file):
            raise ValueError('%s already exists' % out_file)
        num_sample = int(len(train_data) * percent / 100)
        out_data = random.sample(train_data, num_sample)
        with open(out_file, 'w') as f_o:
            for line in out_data:
                f_o.write(line)

def get_train_num(dataset, num_sample):
    data_file = '../data/%s/coreset/train_data_percent_5.jsonl' % dataset
    train_data = read_data(data_file)
    out_file = '../data/%s/coreset/train_data_p_5_num_%d.jsonl' % (dataset, num_sample)
    if os.path.isfile(out_file):
        raise ValueError('%s already exists' % out_file)
    out_data = random.sample(train_data, num_sample)
    with open(out_file, 'w') as f_o:
        for line in out_data:
            f_o.write(line)

def get_shapley_val_data():
    data_file = './open_domain_data/%s/coreset/dev_data.jsonl' % dataset
    out_file = './open_domain_data/%s/coreset/dev_shapley.jsonl' % dataset
    if os.path.isfile(out_file):
        raise ValueError('%s already exists' % out_file)
    dev_data = []
    with open(data_file) as f:
        for line in f:
            dev_data.append(line)
    num_sample = 30
    dev_shapley = random.sample(dev_data, num_sample)
    with open(out_file, 'w') as f_o:
        for line in dev_shapley:
            f_o.write(line)

def add_question_id():
    data_file = './open_domain_data/%s/coreset/train_data_percent_100.jsonl' % dataset
    out_file = './open_domain_data/%s/coreset/train_data_percent_100_updated.jsonl' % dataset
    f_o = open(out_file, 'w') 
    with open(data_file) as f:
        for qid, line in enumerate(f):
            item = json.loads(line)
            item['qid'] = 'train_%d' % qid 
            f_o.write(json.dumps(item) + '\n')
    f_o.close()

def sample_shapley_train_data():
    data_file = './open_domain_data/%s/coreset/train_data_percent_100.jsonl' % dataset
    out_file = './open_domain_data/%s/coreset/train_data_shapley.jsonl' % dataset
    if os.path.isfile(out_file):
        raise ValueError('%s already exists' % out_file)
    
    train_data = read_data(data_file)
    num_shapley = 1000
    train_shapley_data = random.sample(train_data, num_shapley)
    with open(out_file, 'w') as f_o:
        for line in train_shapley_data:
            f_o.write(line)

if __name__ == '__main__':
    get_train_num('NQ', 3010)
    #get_train_percent('TQA')
    #main_top_10('TQA')
    #add_question_id()    
    #get_shapley_val_data()
    #sample_shapley_train_data() 
    
