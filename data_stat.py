import json
import glob



def read_unforgetting(step):
    data = []
    data_file = './output/forgetting/train_5/forgetting_step_%d.jsonl' % step
    with open(data_file) as f:
        for line in f:
            item = json.loads(line)
            qid = item['qid']
            prev_acc = int(item['prev_acc'])
            forgetting = int(item['forgetting'])
            if (not forgetting) and prev_acc:
                data.append(item)

    data_sorted = sorted(data, key=lambda a: a['first_correct_step'])
    with open('./output/forgetting/train_5/unforgetting.jsonl', 'w') as f_o:
        for item in data_sorted:
            f_o.write(json.dumps(item) + '\n')
    return data


def gen_coreset(coreset_tag):
    data_file = '../data/NQ/coreset/train_data_percent_5.jsonl'
    qid_set = set()
    unforgetting_file = './output/forgetting/train_5/unforgetting.jsonl'
    with open(unforgetting_file) as f:
        row = 0
        for line in f:
            item = json.loads(line)
            qid_set.add(item['qid'])
            row += 1
            if row >= 300:
                break
    
    out_file = './output/forgetting/train_5/coreset/train_5_coreset_%s.jsonl' % coreset_tag
    f_o = open(out_file, 'w')
    with open(data_file) as f_2:
        for line in f_2:
            item = json.loads(line)
            if item['qid'] not in qid_set:
                f_o.write(line)
    f_o.close()

def main():
    #read_unforgetting(5937)
    gen_coreset('new_3')

if __name__ == '__main__':
    main()
