import json

def stat_forgetting():
    data_unforgetting = read_unforgetting(11874)
    for step in [9895, 7916, 5937, 3958, 1979]:
        step_unforgetting = read_unforgetting(step)
        for qid in step_unforgetting:
            if qid in data_unforgetting:
                data_unforgetting[qid]['step'] = step
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

def main():
    stat_forgetting() 

if __name__ == '__main__':
    main()
