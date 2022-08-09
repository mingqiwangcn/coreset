import json
import glob


def gen_forgetting_data(step):
    data_learnable = []
    data_unlearnable = []
    data_file = './output/forgetting/train_5/forgetting_step_%d.jsonl' % step
    with open(data_file) as f:
        for line in f:
            item = json.loads(line)
            if item['first_correct_step'] is not None:
                data_learnable.append(item)
            else:
                data_unlearnable.append(item)

    sort_key = lambda a: (a['forgetting'], a['first_correct_step'])
    data_learnable_sorted = sorted(data_learnable, key=sort_key)
    
    out_data = data_learnable_sorted + data_unlearnable
    with open('./output/forgetting/train_5/forgetting_sorted.jsonl', 'w') as f_o:
        for item in out_data:
            f_o.write(json.dumps(item) + '\n')


def gen_coreset(coreset_tag, up_to_rows, strategy_func):
    data_file = '../data/NQ/coreset/train_data_percent_5.jsonl'
    data = []
    forgetting_file = './output/forgetting/train_5/forgetting_sorted.jsonl'
    with open(forgetting_file) as f:
        for line in f:
            item = json.loads(line)
            data.append(item)
    
    out_qid_set = strategy_func(data)
    out_file = './output/forgetting/train_5/coreset/train_5_coreset_%s.jsonl' % coreset_tag
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

def main():
    #gen_forgetting_data(11874) 
    gen_coreset('fg_gt_0', None, remove_zero_forgetting)

if __name__ == '__main__':
    main()
