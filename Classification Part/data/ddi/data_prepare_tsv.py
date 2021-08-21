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
                                                                ,"<e21>", "</e21>", "<e22>", "</e22>"]})
else:
    tokenizer = BertTokenizer.from_pretrained(config.pretrained_model_name, do_lower_case=config.do_lower_case)
    tokenizer.add_special_tokens({"additional_special_tokens": ["<e1>", "</e1>", "<e2>", "</e2>","<e11>", "</e11>", "<e12>", "</e12>"
                                                                ,"<e10>", "</e10>", "<e13>", "</e13>","<e20>", "</e20>", "<e23>", "</e23>"
                                                                ,"<e21>", "</e21>", "<e22>", "</e22>"]})




label_path= 'label2id.json'
label2id = json.load(open(label_path, 'r'))
NA_id = label2id['negative']

def data_split(full_list, ratio, shuffle=False):
    """
    Devide the full_list into sublist_1 and sublist_2
    :param full_list: full_data
    :param ratio:     the ratio of sub_dataset
    :param shuffle:   if random
    :return:
    """
    n_total = len(full_list)
    offset = int(n_total * ratio)
    offset = 12841
    if n_total == 0 or offset < 1:
        return [], full_list
    if shuffle:
        print('shuffle')
        random.shuffle(full_list)
    sublist_1 = full_list[:12098]+full_list[15119:]  #1:[3060:] [:3036] 2:[3060:6056] 3:[6056:9077] 4:[9077:12098] 5:[12098:15119]
    sublist_2 = full_list[12098:15119]
    print(len(sublist_1))
    return sublist_1, sublist_2

def read_tsv(input_file, quotechar=None, split=False):
    """Reads a tab separated value file."""
    with open(input_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
        # reader = csv.reader(f)
        lines = []
        for line in reader:
            lines.append(line)
        if split:
            train_dataset, test_dataset = data_split(lines, 0.77, shuffle=False)
            # print(train_dataset[0])
            # print(test_dataset[0])
            # exit()
            return train_dataset, test_dataset
        else:
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
    for ins in tqdm(data):
        sent_ins = ins[1]
        label = ins[0]
        label = label2id.get(label, NA_id)

        tokenized_text = tokenizer.tokenize(sent_ins)

        # 添加["CLS"]和["SEP"]
        tokenized_text = ["[CLS]"]+tokenized_text

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

        if e11_p > len(e1_mask) - 1 or e12_p > len(e1_mask) - 1 or e21_p > len(e1_mask) - 1 or e22_p > len(e1_mask) - 1:
            drop_accont += 1
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

def get_dataset(file_dir, filename, max_length, split=False):
    if split:
        train_data, test_data = read_tsv(filename, split=split)
        train_ids_pad, train_mask, train_segment, train_labels, train_e1_mask_data, train_e2_mask_data, train_input_length = process_data(
            train_data, max_length)
        train_dataset = TensorDataset(train_ids_pad, train_mask, train_segment, train_labels, train_e1_mask_data, train_e2_mask_data,
                                train_input_length)
        test_ids_pad, test_mask, test_segment, test_labels, test_e1_mask_data, test_e2_mask_data, test_input_length = process_data(
            test_data, max_length)
        test_dataset = TensorDataset(test_ids_pad, test_mask, test_segment, test_labels, test_e1_mask_data,
                                      test_e2_mask_data,
                                      test_input_length)
        # print(train_dataset[0:5])
        # print('-----------------------------')
        # print(test_dataset[0:5])
        return train_dataset, test_dataset

    else:
        data = read_tsv(filename, split=split)
        input_ids_pad, input_mask, input_segment, input_labels, e1_mask_data, e2_mask_data, input_length = process_data(data, max_length)

        dataset = TensorDataset(input_ids_pad, input_mask, input_segment, input_labels, e1_mask_data, e2_mask_data, input_length)
        return dataset


def dump_dataset(data_name, split=False):
    if split:
        file_path = 'five_rounds_test\\'
        train_dataset, test_dataset = get_dataset(file_dir='..', filename=data_name + '.tsv',
                                                  max_length=config.MAX_LENGTH, split=split)
        torch.save(train_dataset, file_path + 'train5' + '.pt')
        torch.save(test_dataset, file_path + 'test5' + '.pt')
    else:
        dataset = get_dataset(file_dir='.', filename=data_name + '.tsv', max_length=config.MAX_LENGTH)
        torch.save(dataset, data_name +'tsv'+'.pt')


# dump_dataset(r'generated data\train_context0.5')
dump_dataset('full_dataset', split=True)
# dump_dataset('test_c')
# dump_dataset("generated data/mechanism_gcv1")
# dump_dataset("generated data/int_gcv1")