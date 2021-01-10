from sklearn import metrics
from torch import optim
from torch.nn.utils import clip_grad_norm_
import torch
from torch.backends import cudnn

import logging
from model import *
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertConfig, RobertaTokenizer, RobertaModel, RobertaConfig
from transformers import AdamW, get_linear_schedule_with_warmup
import os
import torch.nn as nn
import time
from tensorboardX import SummaryWriter
import random
import numpy as np
# from torchsample.callbacks import EarlyStopping

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    cudnn.benchmark = False
    cudnn.enabled = False




logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s %(message)s')

DEVICE = 'cuda:0'
VALID_TIMES = 20

#超参数
hidden_dropout_prob = 0.3
num_labels = 6
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



def train(config, log_path, lr):
    """
    Before training, the input with '.json' format must be transformed into '.pt'
    format by 'data_prepare.py'. This process will also generate the 'vocab.pt'
    file which contains the basic statistics of the corpus.
    """

    writer = SummaryWriter('pic/dti/e-5'+str(lr))
    unfreeze_layer = ['layer.10', 'layer.11', 'bert.pooler', 'seq_multiatten', 'seq_biGRU', 'cls_fc_layer',
                      'e1_fc_layer',
                      'e2_fc_layer', 'label_classifier']
    if config.BERT_MODE==2:
        logging.info('Model: {}'.format(config.pretrained_model_name))
        tokenizer = RobertaTokenizer.from_pretrained(config.pretrained_model_name, do_lower_case=config.do_lower_case)
        tokenizer.add_special_tokens({"additional_special_tokens": ["<e1>", "</e1>", "<e2>", "</e2>"]})
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
        tokenizer.add_special_tokens({"additional_special_tokens": ["<e1>", "</e1>", "<e2>", "</e2>"]})
        bert_config = BertConfig.from_pretrained(config.pretrained_model_name, num_labels=num_labels,
                                                 finetuning_task=config.task)
        model = Mybert_without_entity_information(bert_config, config)
        model.resize_token_embeddings(len(tokenizer))

    if config.BERT_MODE==1:
        logging.info('Model: {}'.format(config.pretrained_model_name))
        tokenizer = BertTokenizer.from_pretrained(config.pretrained_model_name, do_lower_case=config.do_lower_case)
        tokenizer.add_special_tokens({"additional_special_tokens": ["<e1>", "</e1>", "<e2>", "</e2>"]})
        bert_config = BertConfig.from_pretrained(config.pretrained_model_name, num_labels=num_labels,
                                                           finetuning_task=config.task)
        model = Mybert(bert_config, config)
        model.resize_token_embeddings(len(tokenizer))

    if config.BERT_MODE==4:
        logging.info('Model: {}'.format(config.pretrained_model_name))
        logging.info('Model: {}'.format("Mybert_without_sentence_information"))
        tokenizer = BertTokenizer.from_pretrained(config.pretrained_model_name, do_lower_case=config.do_lower_case)
        tokenizer.add_special_tokens({"additional_special_tokens": ["<e1>", "</e1>", "<e2>", "</e2>"]})
        bert_config = BertConfig.from_pretrained(config.pretrained_model_name, num_labels=num_labels,
                                                           finetuning_task=config.task)
        model = Mybert_without_sentence_information(bert_config, config)
        model.resize_token_embeddings(len(tokenizer))
    if config.BERT_MODE==5:
        logging.info('Model: {}'.format(config.pretrained_model_name))
        logging.info('Model: {}'.format("Mybert_without_attn"))
        tokenizer = BertTokenizer.from_pretrained(config.pretrained_model_name, do_lower_case=config.do_lower_case)
        tokenizer.add_special_tokens({"additional_special_tokens": ["<e1>", "</e1>", "<e2>", "</e2>"]})
        bert_config = BertConfig.from_pretrained(config.pretrained_model_name, num_labels=num_labels,
                                                           finetuning_task=config.task)
        model = Mybert_without_attention(bert_config, config)
        model.resize_token_embeddings(len(tokenizer))
    if config.BERT_MODE==6:
        logging.info('Model: {}'.format(config.pretrained_model_name))
        logging.info('Model: {}'.format("Mybert_without_packedBiGRU"))
        tokenizer = BertTokenizer.from_pretrained(config.pretrained_model_name, do_lower_case=config.do_lower_case)
        tokenizer.add_special_tokens({"additional_special_tokens": ["<e1>", "</e1>", "<e2>", "</e2>"]})
        bert_config = BertConfig.from_pretrained(config.pretrained_model_name, num_labels=num_labels,
                                                           finetuning_task=config.task)
        model = Mybert_without_packedBiGRU(bert_config, config)
        model.resize_token_embeddings(len(tokenizer))

    if config.BERT_MODE==7:
        logging.info('Model: {}'.format(config.pretrained_model_name))
        logging.info('Model: {}'.format("Mybert_startent"))
        tokenizer = BertTokenizer.from_pretrained(config.pretrained_model_name, do_lower_case=config.do_lower_case)
        tokenizer.add_special_tokens({"additional_special_tokens": ["<e1>", "</e1>", "<e2>", "</e2>"]})
        bert_config = BertConfig.from_pretrained(config.pretrained_model_name, num_labels=num_labels,
                                                           finetuning_task=config.task)
        model = Mybert_startent(bert_config, config)
        model.resize_token_embeddings(len(tokenizer))

    log_f = open(log_path, 'a')


    # 加载dataloader
    train_dataset = torch.load(os.path.join(config.ROOT_DIR, 'train.pt'))
    train_loader = DataLoader(train_dataset, config.BATCH_SIZE, shuffle=True)

    test_dataset = torch.load(os.path.join(config.ROOT_DIR, 'test.pt'))
    valid_loader = DataLoader(test_dataset, config.BATCH_SIZE,  shuffle=False)

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

    # weight = torch.tensor([0.02, 0.11, 0.05, 0.07, 0.75]) #2:1
    # weight = torch.tensor([0.01, 0.15, 0.08, 0.10, 0.65]) #originial
    # weight = torch.tensor([0.02, 0.15, 0.08, 0.09, 0.66]) #1.5:1
    # weight = torch.tensor([0.01, 0.15, 0.08, 0.10, 0.66])  # pay attention to int and effect
    criterion = nn.CrossEntropyLoss(weight=weight, reduction='mean').to(DEVICE)

    # params = [p for p in model.parameters() if p.requires_grad]
    # optimizer = optim.Adam(params, lr=5e-5, weight_decay=0, eps=1e-8)

    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name, param.size())
    # bert_param_train = list(map(id, [p for p in model.bert.parameters() if p.requires_grad]))
    # bert_param = list(map(id, model.bert.parameters()))
    # bert_param_train_f = filter(lambda p: id(p) in bert_param, model.bert.parameters())
    # base_param = filter(lambda p: id(p) not in bert_param, model.parameters())
    # params = [{"params": base_param, "lr": config.LEARNING_RATE*10,"weight_decay": 1e-8},
    #           {"params": model.bert.parameters(), "lr": config.LEARNING_RATE, "weight_decay": 0}]
    #
    # optimizer = optim.Adam([{"params": base_param, "lr": config.LEARNING_RATE*5,"weight_decay": 0},
    #           {"params": model.bert.parameters(), "lr": config.LEARNING_RATE, "weight_decay": 0}])

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
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.LEARNING_RATE, eps=config.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=t_total)

    model.to(DEVICE)

    validate_every = len(train_loader) // VALID_TIMES

    def run_iter(batch, is_training):
        model.train(is_training)

        if not is_training:
            model.eval()
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

        # print(input_ids[0])
        # s_i = np.array(input_ids.cpu())[0]
        # s_t = tokenizer.convert_ids_to_tokens(s_i)
        # # print(s_t)
        # s = tokenizer.convert_tokens_to_string(s_t)
        # print(s)
        # exit()
        if is_training:
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(parameters=model.parameters(), max_norm=5)
            optimizer.step()
            scheduler.step()

        return loss, label_pred.cpu()

    save_dir = os.path.join(config.SAVE_DIR, config.DATA_SET)
    # if log_path=='ddi_e-4.log':
    #     save_dir = os.path.join(config.SAVE_DIR, "ddi_e-4")
    # if log_path=='ddi_e-5.log':
    #     save_dir = os.path.join(config.SAVE_DIR, "ddi_e-5")
    # if log_path=='log/without_entity.log':
    #     save_dir = os.path.join("checkpoint/BioBert/withoutentity")
    #     save_dir = os.path.join(config.SAVE_DIR, config.DATA_SET)
    # if log_path=='BioRoBerta_ddi_e-4.log':
    #     save_dir = os.path.join(config.SAVE_DIR, "ddi_e-4")

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    best_f1 = 0

    for epoch_num in range(config.MAX_EPOCHS):
        logging.info('Epoch {}: start'.format(epoch_num))

        train_labels = []
        train_preds = []

        valid_loss_sum_full = 0

        train_loss_sum = 0

        valid_labels_full = []
        valid_preds_full = []

        for batch_iter, train_batch in enumerate(train_loader):
            # print(len(train_loader))
            # exit()
            # train_loss, train_pred = run_iter(batch=train_batch, is_training=True)
            # try:
            train_loss, train_pred = run_iter(batch=train_batch, is_training=True)
            # except:
            #     print("fucking windows:train")
            #     continue

            train_loss_sum += train_loss.item()
            # 添加label
            train_labels.extend(train_batch[3])
            # 添加预测值
            train_preds.extend(train_pred)



            # if (batch_iter + 1) % validate_every == 0:
            #
            #     torch.set_grad_enabled(False)
            #
            #     valid_loss_sum = 0
            #
            #     valid_labels = []
            #     valid_preds = []
            #
            #     with torch.no_grad():
            #         for valid_batch in valid_loader:
            #             try:
            #                 valid_loss, valid_pred = run_iter(batch=valid_batch, is_training=False)
            #             except:
            #                 print("fucking windows:test")
            #                 continue
            #
            #             # valid_loss, valid_pred = run_iter(batch=valid_batch, is_training=False)
            #
            #             valid_loss_sum += valid_loss.item()
            #
            #             valid_labels.extend(valid_batch[3])
            #             valid_preds.extend(valid_pred)
            #
            #     torch.set_grad_enabled(True)
            #
            #     valid_loss = valid_loss_sum / len(valid_loader)
            #     # print("test")
            #     # print(valid_loss)
            #     writer.add_scalar('Train/Loss', train_loss, epoch_num * len(train_loader) + batch_iter + 1)
            #     writer.add_scalar('Valid/Loss', valid_loss, epoch_num * len(train_loader) + batch_iter + 1)
            #     valid_p, valid_r, valid_f1, _ = metrics.precision_recall_fscore_support(valid_labels, valid_preds,
            #                                                                             labels=[1, 2, 3, 4],
            #                                                                             average='micro')
            #     valid_accu = metrics.accuracy_score(valid_labels,valid_preds)
            #
            #     writer.add_scalar('Valid/Accu', valid_accu, epoch_num * len(train_loader) + batch_iter + 1)
            #     train_f1 = metrics.f1_score(train_labels, train_preds, [1, 2, 3, 4], average='micro')
            #
            #     train_accu = metrics.accuracy_score(train_labels,train_preds)
            #     writer.add_scalar('Train/Accu', train_accu, epoch_num * len(train_loader) + batch_iter + 1)
            #     progress = epoch_num + (batch_iter + 1) / len(train_loader)
            #
            #     logging.info(
            #         'Epoch {:.2f}: train loss = {:.4f}, train f1 = {:.4f}, test loss = {:.4f}, test f1 = {:.4f}, test auc = {:.4f}'.format(
            #             progress, train_loss, train_f1, valid_loss, valid_f1, valid_accu))
            #
            #     # 选择更好的f1值的model
            #     if valid_f1 > best_f1:
            #         best_f1 = valid_f1
            #         model_filename = ('{}-{:.4f}.pkl'.format(config.DATA_SET, valid_f1))
            #         model_path = os.path.join(save_dir, model_filename)
            #         if best_f1<0.83:
            #             print(r'Get the new best model of F1: {}, but do not save it because F1 is less than 0.83'.format(best_f1))
            #         else:
            #             torch.save(model.state_dict(), model_path)
            #             print('Saved the new best model to {}'.format(model_path))
            #
            #         log_f.write('{}\tlr={}\n'.format(model_filename, config.LEARNING_RATE))
            #         log_f.flush()
                # if batch_iter+80 > len(train_loader):
                #     model_filename = ('{}-{:.4f}-{}.pkl'.format(config.DATA_SET, valid_f1, epoch_num))
                #     model_path = os.path.join(save_dir, model_filename)
                #     torch.save(model.state_dict(), model_path)
                #     print('Saved the  model to {} in {}'.format(model_path,epoch_num))
                #     log_f.write('{}\tlr={}\tepoch_num]{}\n'.format(model_filename, config.LEARNING_RATE, epoch_num))
                #     log_f.flush()

        train_loss = train_loss_sum/len(train_loader)

        with torch.no_grad():
            for valid_batch in valid_loader:
                try:
                    valid_loss, valid_pred = run_iter(batch=valid_batch, is_training=False)
                except:
                    print("fucking windows:test")
                    continue

                # valid_loss, valid_pred = run_iter(batch=valid_batch, is_training=False)

                valid_loss_sum_full += valid_loss.item()

                valid_labels_full.extend(valid_batch[3])
                valid_preds_full.extend(valid_pred)

        # with torch.no_grad():
        #     for valid_batch in valid_loader:
        #         try:
        #             valid_loss, valid_pred = run_iter(batch=valid_batch, is_training=False)
        #         except:
        #             print("fucking windows:test")
        #             continue
        #
        #         # valid_loss, valid_pred = run_iter(batch=valid_batch, is_training=False)
        #
        #         valid_loss_sum_full += valid_loss.item()
        #
        #         valid_labels_full.extend(valid_batch[3])
        #         valid_preds_full.extend(valid_pred)

        torch.set_grad_enabled(True)

        valid_loss = valid_loss_sum_full / len(valid_loader)

        writer.add_scalar('Train/Loss', train_loss, epoch_num)
        writer.add_scalar('Valid/Loss', valid_loss, epoch_num)
        valid_p_f, valid_r_f, valid_f1_f, _ = metrics.precision_recall_fscore_support(valid_labels_full,
                                                                                      valid_preds_full,
                                                                                      labels=[1, 2, 3, 4],
                                                                                      average='micro')
        valid_accu_f = metrics.accuracy_score(valid_labels_full, valid_preds_full)

        writer.add_scalar('Valid/Accu', valid_accu_f, epoch_num)
        train_f1 = metrics.f1_score(train_labels, train_preds, [1, 2, 3, 4], average='micro')

        train_accu = metrics.accuracy_score(train_labels, train_preds)
        writer.add_scalar('Train/Accu', train_accu, epoch_num)


        logging.info(
            'Epoch {}: train loss = {:.4f}, train f1 = {:.4f}, test loss = {:.4f}, test f1 = {:.4f}, test auc = {:.4f}'.format(
                epoch_num, train_loss, train_f1, valid_loss, valid_f1_f, valid_accu_f))


        model_filename = ('{}-{}-{:.4f}.pkl'.format(config.DATA_SET, epoch_num, valid_f1_f))
        model_path = os.path.join(save_dir, model_filename)

        if valid_f1_f > best_f1:
            best_f1 = valid_f1_f
            model_filename = ('{}-{}-{:.4f}.pkl'.format(config.DATA_SET, epoch_num, valid_f1_f))
            model_path = os.path.join(save_dir, model_filename)
            if valid_f1_f < 0.65:
                print(r'Get the new best model of F1: {}, but do not save it because F1 is less than 0.65'.format(
                    valid_f1_f))
            else:
                torch.save(model.state_dict(), model_path)
                print('Saved the new best model to {}'.format(model_path))

        log_f.write('{}\tlr={}\n'.format(model_filename, config.LEARNING_RATE))
        log_f.flush()







    return best_f1


if __name__ == '__main__':
    from data.dti import config

    seed = int(time.time() * 256)
    # seed = 2020
    # print('seed:{}'.format(seed))
    # set_seed(seed)

    log_t = open('log/dti/e-5_time.log', 'a')
    for lr in range(1, 4):
        start = time.time()
        config.LEARNING_RATE = lr / 100000.0
        config.log()
        # try:
        F = train(config, 'log/dti/e-5.log', lr)
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