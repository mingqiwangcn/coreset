import json
import glob

def get_stat_steps():
    file_pattern = './output/forgetting/train_5/forgetting_step_*.jsonl'
    file_lst = glob.glob(file_pattern)
    step_lst = []
    for data_file in file_lst:
        pos_1 = data_file.index('_step_')
        pos_2 = data_file.index('.jsonl')
        offset = pos_1 + len('_step_')
        step = int(data_file[offset:pos_2])
        step_lst.append(step)
    return step_lst

def stat_unforgetting():
    data_unforgetting = {}
    step_lst = get_stat_steps()
    for step in step_lst:
        step_unforgetting = read_unforgetting(step)
        for qid in step_unforgetting:
            if qid not in data_unforgetting:
                data_unforgetting[qid] = {'step':step_unforgetting[qid]['step']}
            else:
                if step_unforgetting[qid]['step'] < data_unforgetting[qid]['step']:
                    data_unforgetting[qid]['step'] = step_unforgetting[qid]['step']

    unforgetting_lst = []
    for qid in data_unforgetting:
        item = {
            'qid':qid,
            'step':data_unforgetting[qid]['step']
        }
        unforgetting_lst.append(item)
    unforgetting_sorted = sorted(unforgetting_lst, key=lambda a: a['step']) 

    unforgetting_step_dict = {}
    for item in unforgetting_sorted:
        step = item['step']
        if step not in unforgetting_step_dict:
            unforgetting_step_dict[step] = []
        unforgetting_step_dict[step].append(item)

    for step in unforgetting_step_dict:
        out_file = './output/forgetting/train_5/unforgetting_from_step_%d.jsonl' % step
        with open(out_file, 'w') as f_o:
            item_lst = unforgetting_step_dict[step]
            for item in item_lst:
                f_o.write(json.dumps(item) + '\n')

def read_unforgetting(step):
    data = {}
    data_file = './output/forgetting/train_5/forgetting_step_%d.jsonl' % step
    with open(data_file) as f:
        for line in f:
            item = json.loads(line)
            qid = item['qid']
            prev_acc = int(item['prev_acc'])
            forgetting = int(item['forgetting'])
            if (not forgetting) and prev_acc:
                data[qid] = {
                    'step':step
                }
    return data

def gen_coreset(step_lst, coreset_tag):
    data_file = '../fusion_reader/open_domain_data/NQ/coreset/train_data_percent_5.jsonl'
    qid_set = set()
    for step in step_lst:
        unforgetting_file = './output/forgetting/train_5/unforgetting_from_step_%d.jsonl' % step
        with open(unforgetting_file) as f:
            for line in f:
                item = json.loads(line)
                qid_set.add(item['qid'])
    
    out_file = './output/forgetting/train_5/coreset/train_5_coreset_%s.jsonl' % coreset_tag
    f_o = open(out_file, 'w')
    with open(data_file) as f_2:
        for line in f_2:
            item = json.loads(line)
            if item['qid'] not in qid_set:
                f_o.write(line)
                   
    f_o.close()

def main():
    stat_unforgetting() 
    gen_coreset([1979, 3958], '2')

if __name__ == '__main__':
    main()
