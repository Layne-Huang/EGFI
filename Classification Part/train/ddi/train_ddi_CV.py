from sklearn import metrics
from torch import optim
from torch.nn.utils import clip_grad_norm_

import logging
from model import *
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertConfig, RobertaTokenizer, RobertaModel, RobertaConfig
from transformers import AdamW, get_linear_schedule_with_warmup
import os
import torch.nn as nn
import time
from tensorboardX import SummaryWriter
import sys
sys.path.append('../../')
from utils import get_k_fold_data
# from torchsample.callbacks import EarlyStopping




logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s %(message)s')

DEVICE = 'cuda:0'
VALID_TIMES = 20

#超参数
hidden_dropout_prob = 0.3
num_labels = 5
learning_rate = 1e-5
weight_decay = 1e-2
epochs = 2
batch_size = 16

# model_name = 'bert-base-uncased'
# MODEL_PATH = r'D:\programming\BERT\Pretrained_models\bert-base-uncased'
#
# # b. 导入配置文件
# model_config = BertConfig.from_pretrained(model_name)
# # 修改配置
# model_config.output_hidden_states = True
# model_config.output_attentions = True



def train(config, log_path, lr, k):
    """
    Before training, the input with '.json' format must be transformed into '.pt'
    format by 'data_prepare.py'. This process will also generate the 'vocab.pt'
    file which contains the basic statistics of the corpus.
    """

    writer = SummaryWriter('../../pic/drugmask_addClassifiedContext0.5'+str(lr))
    unfreeze_layer = ['layer.10', 'layer.11', 'bert.pooler', 'seq_multiatten', 'seq_biGRU', 'cls_fc_layer',
                      'e1_fc_layer',
                      'e2_fc_layer', 'label_classifier']
    if config.BERT_MODE==2:
        logging.info('Model: {}'.format(config.pretrained_model_name))
        tokenizer = RobertaTokenizer.from_pretrained(config.pretrained_model_name, do_lower_case=config.do_lower_case)
        tokenizer.add_special_tokens({"additional_special_tokens": ["<e1>", "</e1>", "<e2>", "</e2>","<e11>", "</e11>", "<e12>", "</e12>"
                                                                ,"<e10>", "</e10>", "<e13>", "</e13>","<e20>", "</e20>", "<e23>", "</e23>"
                                                                ,"<e21>", "</e21>", "<e22>", "</e22>", "drug1", "drug2"]})
        bert_config = RobertaConfig.from_pretrained(config.pretrained_model_name, num_labels=num_labels,
                                                           finetuning_task=config.task)
        model = MyRoberta(bert_config, config)

        model.resize_token_embeddings(len(tokenizer))
        # for name, param in model.named_parameters():
        #     param.requires_grad = False
        #     for ele in unfreeze_layer:
        #         if ele in name:
        #             param.requires_grad = True
        #             break

        # writer.add_graph(model)
    if config.BERT_MODE==3:
        logging.info('Model: {}'.format(config.pretrained_model_name))
        logging.info('Model: {}'.format("Mybert_without_entity_information"))
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

    if config.BERT_MODE==4:
        logging.info('Model: {}'.format(config.pretrained_model_name))
        logging.info('Model: {}'.format("Mybert_without_sentence_information"))
        tokenizer = BertTokenizer.from_pretrained(config.pretrained_model_name, do_lower_case=config.do_lower_case)
        tokenizer.add_special_tokens({"additional_special_tokens": ["<e1>", "</e1>", "<e2>", "</e2>","<e11>", "</e11>", "<e12>", "</e12>"
                                                                ,"<e10>", "</e10>", "<e13>", "</e13>","<e20>", "</e20>", "<e23>", "</e23>"
                                                                ,"<e21>", "</e21>", "<e22>", "</e22>"]})
        bert_config = BertConfig.from_pretrained(config.pretrained_model_name, num_labels=num_labels,
                                                           finetuning_task=config.task)
        model = Mybert_without_sentence_information(bert_config, config)
        model.resize_token_embeddings(len(tokenizer))
    if config.BERT_MODE==5:
        logging.info('Model: {}'.format(config.pretrained_model_name))
        logging.info('Model: {}'.format("Mybert_without_attn"))
        tokenizer = BertTokenizer.from_pretrained(config.pretrained_model_name, do_lower_case=config.do_lower_case)
        tokenizer.add_special_tokens({"additional_special_tokens": ["<e1>", "</e1>", "<e2>", "</e2>","<e11>", "</e11>", "<e12>", "</e12>"
                                                                ,"<e10>", "</e10>", "<e13>", "</e13>","<e20>", "</e20>", "<e23>", "</e23>"
                                                                ,"<e21>", "</e21>", "<e22>", "</e22>"]})
        bert_config = BertConfig.from_pretrained(config.pretrained_model_name, num_labels=num_labels,
                                                           finetuning_task=config.task)
        model = Mybert_without_attention(bert_config, config)
        model.resize_token_embeddings(len(tokenizer))
    if config.BERT_MODE==6:
        logging.info('Model: {}'.format(config.pretrained_model_name))
        logging.info('Model: {}'.format("Mybert_without_packedBiGRU"))
        tokenizer = BertTokenizer.from_pretrained(config.pretrained_model_name, do_lower_case=config.do_lower_case)
        tokenizer.add_special_tokens({"additional_special_tokens": ["<e1>", "</e1>", "<e2>", "</e2>","<e11>", "</e11>", "<e12>", "</e12>"
                                                                ,"<e10>", "</e10>", "<e13>", "</e13>","<e20>", "</e20>", "<e23>", "</e23>"
                                                                ,"<e21>", "</e21>", "<e22>", "</e22>"]})
        bert_config = BertConfig.from_pretrained(config.pretrained_model_name, num_labels=num_labels,
                                                           finetuning_task=config.task)
        model = Mybert_without_packedBiGRU(bert_config, config)
        model.resize_token_embeddings(len(tokenizer))

    if config.BERT_MODE==7:
        logging.info('Model: {}'.format(config.pretrained_model_name))
        logging.info('Model: {}'.format("Mybert_startent"))
        tokenizer = BertTokenizer.from_pretrained(config.pretrained_model_name, do_lower_case=config.do_lower_case)
        tokenizer.add_special_tokens({"additional_special_tokens": ["<e1>", "</e1>", "<e2>", "</e2>","<e11>", "</e11>", "<e12>", "</e12>"
                                                                ,"<e10>", "</e10>", "<e13>", "</e13>","<e20>", "</e20>", "<e23>", "</e23>"
                                                                ,"<e21>", "</e21>", "<e22>", "</e22>", "drug1", "drug2"]})
        bert_config = BertConfig.from_pretrained(config.pretrained_model_name, num_labels=num_labels,
                                                           finetuning_task=config.task)
        model = Mybert_startent(bert_config, config)
        model.resize_token_embeddings(len(tokenizer))

    log_f = open(log_path, 'a')


    # 加载dataloader
    train_dataset = torch.load(os.path.join(config.ROOT_DIR, 'generated data/train_context0.5.pt'))
    train_loader_lenth = 13285
    # train_loader = DataLoader(train_dataset, config.BATCH_SIZE, shuffle=True)

    test_dataset = torch.load(os.path.join(config.ROOT_DIR, 'test_c.pt'))
    test_loader = DataLoader(test_dataset, config.BATCH_SIZE,  shuffle=False)

    logging.info('Number of train pair: {}'.format(len(train_dataset)))
    logging.info('Number of test pair: {}'.format(len(test_dataset)))

    # 创建模型
    # model = REModel(device=DEVICE, model_name=model_name, MODEL_PATH=MODEL_PATH)
    # bert_config = BertConfig.from_pretrained("bert-base-uncased", num_labels=num_labels,
    #                                     )

    # model = BertForSequenceClassification.from_pretrained("bert-large-uncased", config=bert_config)
    # model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels = 5)

    logging.info('Using device {}'.format(DEVICE))



    weight = torch.FloatTensor(config.LOSS_WEIGHT) if config.LOSS_WEIGHT else None

    weight = torch.tensor([0.02, 0.11, 0.05, 0.07, 0.75]) #2:1
    weight = torch.tensor([0.01, 0.15, 0.08, 0.10, 0.65]) #originial
    weight = torch.tensor([0.02, 0.15, 0.08, 0.09, 0.66]) #1.5:1
    weight = torch.tensor([0.01, 0.15, 0.08, 0.10, 0.66])  # pay attention to int and effect
    criterion = nn.CrossEntropyLoss(weight=weight, reduction='mean').to(DEVICE)

    model.to(DEVICE)



    def run_iter(batch, is_training):
        model.train(is_training)
        if is_training:
            model.zero_grad()

        input_ids = batch[0].to(DEVICE)
        attention_mask = batch[1].to(DEVICE)
        token_type_ids = batch[2].to(DEVICE)
        label = batch[3].to(DEVICE)
        e1_mask = batch[4].to(DEVICE)
        e2_mask = batch[5].to(DEVICE)
        length = batch[6].to(DEVICE)

        logits = model(input_ids, attention_mask, token_type_ids, label, e1_mask, e2_mask, length)

        loss = criterion(input=logits, target=label)

        label_pred = logits.max(1)[1]

        if is_training:
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(parameters=model.parameters(), max_norm=5)
            optimizer.step()
            scheduler.step()

        return loss, label_pred.cpu()

    save_dir = os.path.join("../../"+config.SAVE_DIR, config.DATA_SET)

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    train_ls = [0] * (VALID_TIMES+1)
    train_f1 = [0] * (VALID_TIMES+1)
    train_auc = [0] * (VALID_TIMES+1)
    valid_ls = [0] * (VALID_TIMES+1)
    valid_ls = [0] * (VALID_TIMES+1)
    valid_f1 = [0] * (VALID_TIMES+1)
    valid_auc = [0] * (VALID_TIMES+1)

    best_f1 = 0
    for i in range(k):
        logging.info('The {}th fold: start'.format(i))
        for epoch_num in range(config.MAX_EPOCHS):
            logging.info('Epoch {}: start'.format(epoch_num))

            train_labels = []
            train_preds = []

            data_train, data_valid = get_k_fold_data(k, i, train_dataset)
            train_loader = DataLoader(data_train, config.BATCH_SIZE, shuffle=True)
            valid_loader = DataLoader(data_valid, config.BATCH_SIZE, shuffle=True)

            validate_every = len(train_loader) // VALID_TIMES

            if config.max_steps > 0:
                t_total = config.max_steps
                config.MAX_EPOCHS = config.max_steps // (
                        len(train_loader) // config.gradient_accumulation_steps) + 1
            else:
                t_total = len(train_loader) // config.gradient_accumulation_steps * config.MAX_EPOCHS

            print(t_total)
            no_decay = ['bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                 'weight_decay': 0.1},
                {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                 'weight_decay': 0.0}
            ]
            optimizer = AdamW(optimizer_grouped_parameters, lr=config.LEARNING_RATE, eps=config.adam_epsilon)
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=t_total)


            for batch_iter, train_batch in enumerate(train_loader):
                try:
                    train_loss, train_pred = run_iter(batch=train_batch, is_training=True)
                except:
                    print("fucking windows:train")
                    continue

                # 添加label
                train_labels.extend(train_batch[3])
                # 添加预测值
                train_preds.extend(train_pred)

                if (batch_iter + 1) % validate_every == 0:

                    torch.set_grad_enabled(False)

                    valid_loss_sum = 0
                    valid_labels = []
                    valid_preds = []

                    test_loss_sum = 0
                    test_labels = []
                    test_preds = []
                    with torch.no_grad():
                        for valid_batch in valid_loader:
                            try:
                                valid_loss, valid_pred = run_iter(batch=valid_batch, is_training=False)
                            except:
                                print("fucking windows:test")
                                continue
                            # valid_loss, valid_pred = run_iter(batch=valid_batch, is_training=False)
                            valid_loss_sum += valid_loss.item()
                            valid_labels.extend(valid_batch[3])
                            valid_preds.extend(valid_pred)
                        for test_batch in test_loader:
                            try:
                                test_loss, test_pred = run_iter(batch=test_batch, is_training=False)
                            except:
                                print("fucking windows:test")
                                continue
                            test_loss_sum += test_loss.item()
                            test_labels.extend(test_batch[3])
                            test_preds.extend(test_pred)

                    torch.set_grad_enabled(True)

                    valid_loss = valid_loss_sum / len(valid_loader)
                    test_loss = test_loss_sum / len(test_loader)

                    writer.add_scalar('Train/Loss', train_loss, epoch_num * len(train_loader) + batch_iter + 1)
                    writer.add_scalar('Valid/Loss', valid_loss, epoch_num * len(train_loader) + batch_iter + 1)
                    valid_p, valid_r, valid_f1, _ = metrics.precision_recall_fscore_support(valid_labels, valid_preds,
                                                                                            labels=[1, 2, 3, 4],
                                                                                            average='micro')
                    valid_accu = metrics.accuracy_score(valid_labels,valid_preds)

                    writer.add_scalar('Valid/Accu', valid_accu, epoch_num * len(train_loader) + batch_iter + 1)
                    train_f1 = metrics.f1_score(train_labels, train_preds, [1, 2, 3, 4], average='micro')

                    train_accu = metrics.accuracy_score(train_labels,train_preds)
                    writer.add_scalar('Train/Accu', train_accu, epoch_num * len(train_loader) + batch_iter + 1)

                    test_f1 = metrics.f1_score(test_labels, test_preds, [1, 2, 3, 4], average='micro')
                    test_accu = metrics.accuracy_score(test_labels, test_preds)
                    progress = epoch_num + (batch_iter + 1) / len(train_loader)

                    logging.info(
                        'Epoch {:.2f}: train loss = {:.4f}, train f1 = {:.4f}, valid loss = {:.4f}, valid f1 = {:.4f}, valid auc = {:.4f} test loss = {:.4f}, test f1 = {:.4f}, test auc = {:.4f}'.format(
                            progress, train_loss, train_f1, valid_loss, valid_f1, valid_accu, test_loss, test_f1, test_accu))

                    # 选择更好的f1值的model
                    if valid_f1 > best_f1:
                        best_f1 = valid_f1
                        model_filename = ('{}-{:.4f}.pkl'.format(config.DATA_SET, valid_f1))
                        model_path = os.path.join(save_dir, model_filename)
                        if best_f1<0.83:
                            print(r'{} fold: Get the new best model of valid F1: {}, test F1:{}, but do not save it because F1 is less than 0.83'.format(i, best_f1, test_f1))
                        else:
                            torch.save(model.state_dict(), model_path)
                            print('{} fold: Saved the new best model (valid F1: {}; test F1: {}) to {}'.format(i, best_f1, test_f1, model_path))

                        log_f.write('{}\tlr={}\ttest_f1={}\n'.format(model_filename, config.LEARNING_RATE, test_f1))
                        log_f.flush()
                    # if batch_iter+80 > len(train_loader):
                    #     model_filename = ('{}-{:.4f}-{}.pkl'.format(config.DATA_SET, valid_f1, epoch_num))
                    #     model_path = os.path.join(save_dir, model_filename)
                    #     torch.save(model.state_dict(), model_path)
                    #     print('Saved the  model to {} in {}'.format(model_path,epoch_num))
                    #     log_f.write('{}\tlr={}\tepoch_num]{}\n'.format(model_filename, config.LEARNING_RATE, epoch_num))
                    #     log_f.flush()


    return best_f1

def evaluate(loss_sum, labels, preds):


if __name__ == '__main__':
    from data.ddi import config

    # sys.path.append('../../')

    log_t = open('../../log/drugmask_addClassifiedContext0.5_time.log', 'a')
    for lr in range(1, 6):
        start = time.time()
        config.LEARNING_RATE = lr / 100000.0
        config.log()
        # try:
        F = train(config, '../../log/drugmask_addClassifiedContext0.5_e-5.log', lr, 10)
        end = time.time()
        time_spend = end-start
        log_t.write('spend time={}\tlr={}\n'.format(time_spend, config.LEARNING_RATE))
        log_t.flush()
        # except:
        #     print("fuck windows")
        #     continue








    # for lr in range(1, 11):
    #     start = time.time()
    #     config.LEARNING_RATE = lr / 10000.0
    #     config.log()
    #     F = train(config, 'ddi_e-4.log')
    #     end = time.time()
    #     time_spend = end - start
    #     log_t.write('spend time={}\tlr={}\n'.format(time_spend, config.LEARNING_RATE))
    #     log_t.flush()

    # config.LEARNING_RATE = 1 / 100000.0
    # config.log()
    # F = train(config, 'ddi.log')