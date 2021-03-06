import json
from transformers import BertTokenizer
import random
from torch.utils.data import TensorDataset
import os
import logging
import torch
from tqdm import tqdm
import config

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s %(message)s')

model_name = 'bert-base-uncased'


# a. 通过词典导入分词器
tokenizer = BertTokenizer.from_pretrained(config.pretrained_model_name, do_lower_case=config.do_lower_case)
tokenizer.add_special_tokens({"additional_special_tokens": ["<e1>", "</e1>", "<e2>", "</e2>"]})




label_path= 'label2id.json'
label2id = json.load(open(label_path, 'r'))
NA_id = label2id['NA']

def tokenizer_entity(sent, headword_pos, tailword_pos):
    sent = list(sent)
    if headword_pos[1]<tailword_pos[0]:
        sent.insert(headword_pos[0], "<e1>")
        sent.insert(headword_pos[1]+1, "</e1>")
        sent.insert(tailword_pos[0]+2, "<e2>")
        sent.insert(tailword_pos[1]+3, "</e2>")
    else:
        sent.insert(tailword_pos[0], "<e2>")
        sent.insert(tailword_pos[1] + 1, "</e2>")
        sent.insert(headword_pos[0] + 2, "<e1>")
        sent.insert(headword_pos[1] + 3, "</e1>")
    sent = "".join(sent)
    return sent

def read_data(file_dir, filename):
    data = []
    data_path = os.path.join(file_dir, filename)
    d = json.load(open(data_path, 'r'))
    for ins in d:
        sent = ins['sentence'].replace('\n', '').lower()
        label = label2id.get(ins['relation'], NA_id)
        tail_word = ins['tail']['word'].lower()
        head_word = ins['head']['word'].lower()
        data.append([sent, label,head_word,tail_word])
    random.shuffle(data)
    return data


def process_data(data, max_length):
    def pad(x):
        return x[:max_length] if len(x) > (max_length) else x + [0] * ((max_length) - len(x))
    # sent_raw = [x for x, _, _, _ in data]
    # labels = [y for _, y, _, _ in data]
    # head_word = [h for _, _, h, _ in data]
    # tail_word = [t for _, _, _, t in data]

    input_ids_pad=[]
    input_mask_data=[]
    input_segment_data=[]
    input_labels = []
    # input_labels=[y for _, y, _, _, in data]
    e1_mask_data = []
    e2_mask_data = []
    input_length = []
    drop_accont = 0
    for ins in tqdm(data):
        sent_ins = ins[0]
        lable = ins[1]
        head_word = ins[2]
        tail_word = ins[3]


        # head_pos = sent_ins.index(head_word)
        # head_pos = [head_pos, head_pos + len(head_word)]
        # tail_pos = sent_ins.index(tail_word)
        # tail_pos = [tail_pos, tail_pos + len(tail_word)]
        try:
            head_pos = sent_ins.index(head_word)
            head_pos = [head_pos, head_pos + len(head_word)]
            tail_pos = sent_ins.index(tail_word)
            tail_pos = [tail_pos, tail_pos + len(tail_word)]
        except:
            print(sent_ins)
            print(head_word)
            print(tail_word)
            exit()

        sent_ins = tokenizer_entity(sent_ins,head_pos,tail_pos)
        tokenized_text = tokenizer.tokenize(sent_ins)

        # 添加["CLS"]和["SEP"]
        tokenized_text = ["CLS"]+tokenized_text



        e11_p = tokenized_text.index("<e1>")
        e12_p = tokenized_text.index("</e1>")
        e21_p = tokenized_text.index("<e2>")
        e22_p = tokenized_text.index("</e2>")



        tokenized_text[e11_p] = "$"
        tokenized_text[e12_p] = "$"
        tokenized_text[e21_p] = "#"
        tokenized_text[e11_p] = "#"




        input_ids = tokenizer.convert_tokens_to_ids(tokenized_text)
        length = len(input_ids)
        if length>max_length:
            length = max_length

        input_ids = pad(input_ids)

        input_mask = [1 if i != 0 else 0 for i in input_ids]
        input_segment = [0 for i in input_ids]

        # e1 mask, e2 mask
        e1_mask = [0] * len(input_mask)
        e2_mask = [0] * len(input_mask)

        if e11_p > len(e1_mask)-1or e12_p > len(e1_mask)-1 or e21_p > len(e1_mask)-1 or e22_p > len(e1_mask)-1:
            drop_accont+=1
            continue
        # e1_mask和e2_mask实际上就是对实体所在部分标记为1，其他部分标记为0
        for i in range(e11_p,e12_p+1):
            if i>len(e1_mask)-1:
                print(sent_ins)
                print(tokenized_text)
                print(i)
                print(len(e1_mask))
                exit()
            else:
                e1_mask[i] = 1
        for i in range(e21_p,e22_p+1):
            if i>len(e2_mask)-1:
                print(sent_ins)
                print(tokenized_text)
                print(i)
                print(len(e2_mask))
                exit()
            else:
                e2_mask[i] = 1

        input_ids_pad.append(input_ids)
        input_mask_data.append(input_mask)
        input_segment_data.append(input_segment)
        e1_mask_data.append(e1_mask)
        e2_mask_data.append(e2_mask)
        input_labels.append(lable)
        input_length.append(length)


    input_ids_pad = torch.tensor(input_ids_pad, dtype=torch.long)
    input_mask_data = torch.tensor(input_mask_data, dtype=torch.long)
    input_segment_data = torch.tensor(input_segment_data, dtype=torch.long)
    e1_mask_data = torch.tensor(e1_mask_data, dtype=torch.long)
    e2_mask_data = torch.tensor(e2_mask_data, dtype=torch.long)
    input_labels = torch.tensor(input_labels, dtype=torch.long)
    input_length = torch.tensor(input_length, dtype=torch.long)

    print("Drop number {}".format(drop_accont))

    return input_ids_pad, input_mask_data, input_segment_data, input_labels, e1_mask_data, e2_mask_data, input_length

def get_dataset(file_dir, filename, max_length):
    data = read_data(file_dir,filename)
    print()
    input_ids_pad, input_mask, input_segment, input_labels, e1_mask_data, e2_mask_data, input_length = process_data(data, max_length)
    dataset = TensorDataset(input_ids_pad, input_mask,input_segment, input_labels, e1_mask_data, e2_mask_data, input_length)
    return dataset


def dump_dataset(data_name):
    dataset = get_dataset(file_dir='.', filename=data_name + '.json', max_length=config.MAX_LENGTH)
    torch.save(dataset, data_name + '.pt')


dump_dataset('train')
dump_dataset('valid')
dump_dataset('test')
