import json
from transformers import BertTokenizer, RobertaModel, RobertaTokenizer
import random
from torch.utils.data import TensorDataset
import os
import logging
import torch
from tqdm import tqdm
import config
import csv

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s %(message)s')

model_name = 'bert-base-uncased'


# a. 通过词典导入分词器
if config.BERT_MODE==2:
    tokenizer = RobertaTokenizer.from_pretrained(config.pretrained_model_name, do_lower_case=config.do_lower_case)
    tokenizer.add_special_tokens({"additional_special_tokens": ["<e1>", "</e1>", "<e2>", "</e2>","<e11>", "</e11>", "<e12>", "</e12>"
                                                                ,"<e10>", "</e10>", "<e13>", "</e13>","<e20>", "</e20>", "<e23>", "</e23>"
                                                                ,"<e21>", "</e21>", "<e22>", "</e22>", "drug1", "drug2"]})
else:
    tokenizer = BertTokenizer.from_pretrained(config.pretrained_model_name, do_lower_case=config.do_lower_case)
    tokenizer.add_special_tokens({"additional_special_tokens": ["<e1>", "</e1>", "<e2>", "</e2>","<e11>", "</e11>", "<e12>", "</e12>"
                                                                ,"<e10>", "</e10>", "<e13>", "</e13>","<e20>", "</e20>", "<e23>", "</e23>"
                                                                ,"<e21>", "</e21>", "<e22>", "</e22>", "drug1", "drug2"]})




label_path= 'label2id.json'
label2id = json.load(open(label_path, 'r'))
NA_id = label2id['negative']


def read_tsv(input_file, quotechar=None):
    """Reads a tab separated value file."""
    with open(input_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
        # reader = csv.reader(f)
        lines = []
        for line in reader:
            lines.append(line)
        return lines

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

def find_special(tokenized_text):
    start_e1_tokens = ["<e10>", "<e11>", "<e12>", "<e13>"]
    end_e1_tokens = ["</e10>", "</e11>", "</e12>", "</e13>"]
    start_e2_tokens = ["<e20>", "<e21>", "<e22>", "<e23>"]
    end_e2_tokens = ["</e20>", "</e21>", "</e22>", "</e23>"]

    for x in start_e1_tokens:
        if x in tokenized_text:
            e11_p = tokenized_text.index(x)  # the start position of entity1
            break
    for x in end_e1_tokens:
        if x in tokenized_text:
            e12_p = tokenized_text.index(x)  # the end position of entity1
            break
    for x in start_e2_tokens:
        if x in tokenized_text:
            e21_p = tokenized_text.index(x)  # the start position of entity2
            break
    for x in end_e2_tokens:
        if x in tokenized_text:
            e22_p = tokenized_text.index(x)  # the end position of entity2
            break
    return e11_p, e12_p, e21_p, e22_p

def read_data(file_dir, filename):
    data = []
    data_path = os.path.join(file_dir, filename)
    d = json.load(open(data_path, 'r'))
    for ins in d:
        sent = ins['sentence'].replace('\n', '').lower()
        label = label2id.get(ins['relation'], NA_id)
        tail_word = ins['tail']['word']
        head_word = ins['head']['word']
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
    e1_mask_data = []
    e2_mask_data = []
    input_length = []
    drop_accont = 0
    count=0
    for ins in tqdm(data):
        try:
            sent_ins = ins[1]
            label = ins[0]
        except:
            print(ins)
        label = label2id.get(label, NA_id)




        # # 添加["CLS"]和["SEP"]
        # tokenized_text = ["[CLS]"]+tokenized_text
        try:
            e11_p, e12_p, e21_p, e22_p = find_special(sent_ins)
        except:
            continue

        # print(sent_ins[e11_p+6:e12_p-1])
        drug1 = sent_ins[e11_p+6:e12_p-1]
        drug2 = sent_ins[e21_p + 6:e22_p - 1]
        sent_ins = sent_ins.replace(drug1,'drug1')
        sent_ins = sent_ins.replace(drug2, 'drug2')

        # print(sent_ins)
        # exit()
        tokenized_text = tokenizer.tokenize(sent_ins)
        tokenized_text = ['CLS'] + tokenized_text
        # print(tokenizer.tokenize('[CLS]'))
        # print(tokenizer.convert_tokens_to_ids(tokenizer.tokenize('[CLS]')))
        # print([tokenizer.cls_token])
        # exit()
        # print(tokenized_text)
        try:
            e11_p, e12_p, e21_p, e22_p = find_special(tokenized_text)
        except:
            continue

        # print(tokenized_text[e11_p:e12_p+1])
        # print(tokenized_text)

        input_ids = tokenizer.convert_tokens_to_ids(tokenized_text)
        # if count<5:
        #     print(input_ids)
        #     count+=1
        # else:
        #     exit()

        length = len(input_ids)
        if length>max_length:
            length = max_length

        input_ids = pad(input_ids)

        input_mask = [1 if i != 0 else 0 for i in input_ids]
        input_segment = [0 for i in input_ids]

        # e1 mask, e2 mask
        e1_mask = [0] * len(input_mask)
        e2_mask = [0] * len(input_mask)

        if e11_p > len(e1_mask) - 1 or e12_p > len(e1_mask) - 1 or e21_p > len(e1_mask) - 1 or e22_p > len(e1_mask) - 1:
            drop_accont += 1
            continue

        # e1_mask和e2_mask实际上就是对实体所在部分标记为1，其他部分标记为0
        for i in range(e11_p, e12_p + 1):
            if i > len(e1_mask) - 1:
                print(sent_ins)
                print(tokenized_text)
                print(i)
                print(len(e1_mask))
                exit()
            else:
                e1_mask[i] = 1
        for i in range(e21_p, e22_p + 1):
            if i > len(e2_mask) - 1:
                print(sent_ins)
                print(tokenized_text)
                print(i)
                print(len(e2_mask))
                exit()
            else:
                e2_mask[i] = 1

        # print(e1_mask)
        # exit()

        input_ids_pad.append(input_ids)
        input_mask_data.append(input_mask)
        input_segment_data.append(input_segment)
        e1_mask_data.append(e1_mask)
        e2_mask_data.append(e2_mask)
        input_labels.append(label)
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
    data = read_tsv(filename)
    input_ids_pad, input_mask, input_segment, input_labels, e1_mask_data, e2_mask_data, input_length = process_data(data, max_length)
    # data = [input_ids_pad, input_mask, input_segment, input_labels, e1_mask_data, e2_mask_data, input_length]
    # data.sort(key=lambda a: a[6], reverse=True)
    dataset = TensorDataset(input_ids_pad, input_mask, input_segment, input_labels, e1_mask_data, e2_mask_data, input_length)
    return dataset


def dump_dataset(data_name):
    dataset = get_dataset(file_dir='..', filename=data_name + '.tsv', max_length=config.MAX_LENGTH)
    torch.save(dataset, data_name + '.pt')

# dump_dataset(r'train_c')
# dump_dataset('train_c')
# dump_dataset('test_c')
dump_dataset("generated data/train_context1")
# dump_dataset("generated data/advise_context3c")
# dump_dataset("generated data/effect_context3c")
# dump_dataset("generated data/mechanism_context3c")
# dump_dataset("generated data/int_context3c")