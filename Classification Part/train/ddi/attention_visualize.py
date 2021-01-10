# coding: utf-8
from sklearn import metrics

import logging
import utils as utils
from torch.utils.data import DataLoader
from model import *
from transformers import BertTokenizer, BertConfig, RobertaTokenizer, RobertaConfig
import matplotlib
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import heapq
import csv

def highlight(word, attn):
    html_color = '#%02X%02X%02X' % (255, int(255*(1 - attn)), int(255*(1 - attn)))
    return '<span style="background-color: {}">{}</span>'.format(html_color, word)

def mk_html(seq, attns):
    html = ""
    for ix, attn in zip(seq, attns):
        html += ' ' + highlight(
            ix,
            attn
        )
    return html + "<br>"

from IPython.display import HTML, display
batch_size = 1
seqs = [["这", "是", "一个", "测试", "样例", "而已"]]
attns = [[0.01, 0.19, 0.12, 0.7, 0.2, 0.1]]

for i in range(batch_size):
    text = mk_html(seqs[i], attns[i])
    display(HTML(text))

def softmax(x):
    assert (len(x.shape)==2)
    row_max = np.max(x).reshape(-1,1)
    x -= row_max
    x_exp = np.exp(x)
    s = x_exp / np.sum(x_exp)
    return s

def find_n_sentences(score,sentences,n):
    sentences_n_largest = []
    score_n_largest = heapq.nlargest(n,score)
    for s in score_n_largest:
        i = score.index(s)
        sentences_n_largest.append(sentences[i])
    sentences_n_smallest = []
    score_n_smallest = heapq.nsmallest(n, score)
    for s in score_n_smallest:
        i = score.index(s)
        sentences_n_smallest.append(sentences[i])

    return score_n_largest, sentences_n_largest, score_n_smallest, sentences_n_smallest

#超参数
hidden_dropout_prob = 0.3
num_labels = 5
learning_rate = 1e-5
weight_decay = 1e-2
epochs = 2
batch_size = 16

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s %(message)s')

DEVICE = 'cuda:0'


def test(config, model_name="ddi_e-5-0.9229.pkl"):
    scores = []
    sentences = []
    valid_sentence = []

    lable_error = {1:0, 2:0, 3:0, 4:0 }

    # vocab = torch.load(os.path.join(config.ROOT_DIR, 'vocab.pt'))
    #+++
    # logging.info('Load pretrained vectors: {}*{}'.format(vocab.word_num, vocab.word_dim))
    # logging.info('Number of classes: {}'.format(vocab.class_num))

    if config.BERT_MODE==2:
        logging.info('Model: {}'.format(config.pretrained_model_name))
        tokenizer = RobertaTokenizer.from_pretrained(config.pretrained_model_name, do_lower_case=config.do_lower_case)
        tokenizer.add_special_tokens({"additional_special_tokens": ["<e1>", "</e1>", "<e2>", "</e2>","<e11>", "</e11>", "<e12>", "</e12>"
                                                                ,"<e10>", "</e10>", "<e13>", "</e13>","<e20>", "</e20>", "<e23>", "</e23>"
                                                                ,"<e21>", "</e21>", "<e22>", "</e22>"]})
        bert_config = RobertaConfig.from_pretrained(config.pretrained_model_name, num_labels=num_labels,
                                                           finetuning_task=config.task)
        model = MyRoberta(bert_config, config)
        model.resize_token_embeddings(len(tokenizer))

    if config.BERT_MODE==3:
        logging.info('Model: {}'.format(config.pretrained_model_name))
        tokenizer = BertTokenizer.from_pretrained(config.pretrained_model_name, do_lower_case=config.do_lower_case)
        tokenizer.add_special_tokens({"additional_special_tokens": ["<e1>", "</e1>", "<e2>", "</e2>","<e11>", "</e11>", "<e12>", "</e12>"
                                                                ,"<e10>", "</e10>", "<e13>", "</e13>","<e20>", "</e20>", "<e23>", "</e23>"
                                                                ,"<e21>", "</e21>", "<e22>", "</e22>"]})
        bert_config = BertConfig.from_pretrained(config.pretrained_model_name, num_labels=num_labels,
                                                 finetuning_task=config.task)
        model = Mybert_without_entity_information(bert_config, config)
        model.resize_token_embeddings(len(tokenizer))

    if config.BERT_MODE==1:
        logging.info('Model: {}'.format(config.pretrained_model_name))
        tokenizer = BertTokenizer.from_pretrained(config.pretrained_model_name, do_lower_case=config.do_lower_case)
        tokenizer.add_special_tokens({"additional_special_tokens": ["<e1>", "</e1>", "<e2>", "</e2>","<e11>", "</e11>", "<e12>", "</e12>"
                                                                ,"<e10>", "</e10>", "<e13>", "</e13>","<e20>", "</e20>", "<e23>", "</e23>"
                                                                ,"<e21>", "</e21>", "<e22>", "</e22>", "drug1", "drug2"]})
        bert_config = BertConfig.from_pretrained(config.pretrained_model_name, num_labels=num_labels,
                                                           finetuning_task=config.task)
        model = Mybert(bert_config, config)
        model.resize_token_embeddings(len(tokenizer))

    # test_dataset = torch.load(os.path.join(config.ROOT_DIR, 'generated data/int_gcv1.pt'))
    test_dataset = torch.load(os.path.join(config.ROOT_DIR, 'train_c.pt'))
    test_loader = DataLoader(test_dataset, 1, shuffle=True)

    logging.info('Number of test pair: {}'.format(len(test_dataset)))




    # num_params = sum(np.prod(p.size()) for p in model.parameters())
    # num_embedding_params = np.prod(model.word_emb.weight.size()) + np.prod(model.tag_emb.weight.size())
    # print('# of parameters: {}'.format(num_params))
    # print('# of word embedding parameters: {}'.format(num_embedding_params))
    # print('# of parameters (excluding embeddings): {}'.format(num_params - num_embedding_params))

    if model_name is None:
        model_path = utils.best_model_path(config.SAVE_DIR, config.DATA_SET, i=0)

        logging.info('Loading the best model on validation set: {}'.format(model_path))
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
    else:
        model_path = os.path.join(config.SAVE_DIR, config.DATA_SET, model_name)
        # model_path = r"checkpoint/BioBert\drugmask\addClassifieddata\lossweight\lossweight-0.8207-13.pkl"
        model_path = r"../../checkpoint/BioBert\drugmask\lossweight-0.8472.pkl"
        # model_path = os.path.join('checkpoint/BioBert/biobert_gru2_drop00_ddi_e-5', model_name)
        logging.info('Loading the model: {}'.format(model_path))
        model.load_state_dict(
            torch.load(model_path, map_location='cpu'))
    model.eval()
    model.to(DEVICE)
    # model.display()

    torch.set_grad_enabled(False)

    def run_iter(batch):
        sent = batch[0].to(DEVICE)
        mask = batch[1].to(DEVICE)
        segment = batch[2].to(DEVICE)
        label = batch[3].to(DEVICE)
        e1_mask = batch[4].to(DEVICE)
        e2_mask = batch[5].to(DEVICE)
        length = batch[6].to(DEVICE)
        logits,atten_weights= model(input_ids=sent, attention_mask=mask, token_type_ids=segment, labels=label, e1_mask=e1_mask, e2_mask=e2_mask, length=length)
        # print(logits)
        # print(sent[:length])
        score = softmax(logits.cpu().numpy())
        # print(length)
        # print(atten_weights[0][3][:length])
        # exit()

        scores.append(score[0][1])
        #
        s_i = sent[0][:length]
        s_t = tokenizer.convert_ids_to_tokens(s_i)
        # print(s_t)
        s = tokenizer.convert_tokens_to_string(s_t)
        sentences.append(s.replace(" _ ", "_").replace(" ,", ",").replace(" - ","-"))
        # print(s_t)
        # exit()
        label_pred = logits.max(1)[1]
        if label_pred[0]==1:
            if score[0][1]>=0.99994147:
                print(s_t)
                print(atten_weights[0][1][:length])
                valid_sentence.append(s)


        return label_pred.cpu()

    test_labels = []
    test_preds = []

    for test_batch in test_loader:
        test_pred = run_iter(batch=test_batch)

        test_labels.extend(test_batch[3])
        test_preds.extend(test_pred)


    # a, b, c, d = find_n_sentences(scores, sentences, 300)
    # print(a)
    # print(b)
    # print(c)
    # print(d)

    test_p, test_r, test_f1, _ = metrics.precision_recall_fscore_support(test_labels, test_preds,
                                                                         labels=[1,2,3,4],
                                                                         average='micro')
    test_p_a, test_r_a, test_f1_a, _ = metrics.precision_recall_fscore_support(test_labels, test_preds,
                                                                         labels=[1],
                                                                         average='micro')
    test_p_e, test_r_e, test_f1_e, _ = metrics.precision_recall_fscore_support(test_labels, test_preds,
                                                                         labels=[2],
                                                                         average='micro')
    test_p_m, test_r_m, test_f1_m, _ = metrics.precision_recall_fscore_support(test_labels, test_preds,
                                                                         labels=[3],
                                                                         average='micro')
    test_p_i, test_r_i, test_f1_i, _ = metrics.precision_recall_fscore_support(test_labels, test_preds,
                                                                         labels=[4],
                                                                         average='micro')
    # plt.figure("ROC Curve")
    # plt.title("ROC Curve")
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # precision, recall, _ = metrics.roc_curve(test_labels, test_preds)
    # plt.plot(recall,precision)
    # plt.show()
    # for i, l in enumerate(test_labels):
    #     if l!=test_preds[i] and int(l)!=0:
    #         lable_error[int(l)]+=1

    logging.info(
        'precision =  {:.4f}: recall = {:.4f}, fscore = {:.4f}'.format(test_p, test_r, test_f1))
    logging.info(
        'advise: precision =  {:.4f}: recall = {:.4f}, fscore = {:.4f}'.format(test_p_a, test_r_a, test_f1_a))
    logging.info(
        'effect: precision =  {:.4f}: recall = {:.4f}, fscore = {:.4f}'.format(test_p_e, test_r_e, test_f1_e))
    logging.info(
        'mechanism: precision =  {:.4f}: recall = {:.4f}, fscore = {:.4f}'.format(test_p_m, test_r_m, test_f1_m))
    logging.info(
        'int: precision =  {:.4f}: recall = {:.4f}, fscore = {:.4f}'.format(test_p_i, test_r_i, test_f1_i))
    # print(lable_error)

    a,b,c,d = find_n_sentences(scores,sentences, 10)
    print(a[-1])
    # print(b)
    # print(c)
    # print(d)

    return valid_sentence, b


if __name__ == '__main__':
    from data.ddi import config

    valid_sentence,_ = test(config)

    print(len(valid_sentence))

    print(valid_sentence[0:10][6:])
    # label = 'mechanism'
    # label_gen = open("../../data/ddi/generated data/"+label + "_classified1.tsv", 'w+', newline='', encoding='utf-8')
    # writer = csv.writer(label_gen, delimiter='\t')
    # for sentence in valid_sentence:
    #     sentence = sentence[6:]
    #     writer.writerow([label, sentence])



