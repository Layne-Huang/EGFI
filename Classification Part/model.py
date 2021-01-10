from transformers import BertPreTrainedModel
from transformers import BertModel, RobertaTokenizer, RobertaModel
import torch.nn as nn
import torch
from utils import *
import numpy as np
import torch.nn.functional as F

class FCLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0., use_activation=True):
        super(FCLayer, self).__init__()
        self.use_activation = use_activation
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, output_dim)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.dropout(x)
        if self.use_activation:
            x = self.tanh(x)

        return self.linear(x)

class MultiAttn(nn.Module):
    def __init__(self, in_dim, head_num=8):
        super(MultiAttn, self).__init__()

        self.head_dim = in_dim // head_num
        self.head_num = head_num

        # scaled dot product attention
        self.scale = self.head_dim ** -0.5

        self.w_qs = nn.Linear(in_dim, head_num * self.head_dim, bias=True)
        self.w_ks = nn.Linear(in_dim, head_num * self.head_dim, bias=True)
        self.w_vs = nn.Linear(in_dim, head_num * self.head_dim, bias=True)

        self.w_os = nn.Linear(head_num * self.head_dim, in_dim, bias=True)

        self.gamma = nn.Parameter(torch.FloatTensor([0]))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, attn_mask, non_pad_mask):
        B, L, H = x.size()
        head_num = self.head_num
        head_dim = self.head_dim

        q = self.w_qs(x).view(B * head_num, L, head_dim)
        k = self.w_ks(x).view(B * head_num, L, head_dim)
        v = self.w_vs(x).view(B * head_num, L, head_dim)

        attn_mask = attn_mask.repeat(head_num, 1, 1)

        attn = torch.bmm(q, k.transpose(1, 2))  # B*head_num, L, L
        attn = self.scale * attn
        attn = attn.masked_fill_(attn_mask, -np.inf)
        attn = self.softmax(attn)

        out = torch.bmm(attn, v)  # B*head_num, L, head_dim

        out = out.view(B, L, head_dim * head_num)

        out = self.w_os(out)

        out = non_pad_mask * out

        out = self.gamma * out + x

        return out, attn

class PackedGRU(nn.Module):
    def __init__(self, in_dim, hid_dim, bidirectional=True):
        super(PackedGRU, self).__init__()

        self.gru = nn.GRU(in_dim, hid_dim, num_layers=1, dropout=0.5, batch_first=True, bidirectional=bidirectional)
        # 正交初始化weights
        # nn.init.orthogonal(self.gru.all_weights[0][0])
        # nn.init.orthogonal(self.gru.all_weights[0][1])

    def forward(self, x, length):
        packed = torch.nn.utils.rnn.pack_padded_sequence(x, length, batch_first=True, enforce_sorted=False)
        out, _ = self.gru(packed)
        finalout, _ = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True,total_length=300)

        return finalout

class GlobalMaxPool1d(nn.Module):
    def __init__(self):
        super(GlobalMaxPool1d, self).__init__()
    def forward(self,x):
        return F.max_pool1d(x, kernel_size = x.shape[2])


class textCNN(nn.Module):
    def __init__(self, in_dim, kernel_sizes, nums_channels):
        super(textCNN, self).__init__()
        # self.dropout = nn.Dropout(0.5)
        self.convs = nn.ModuleList()
        self.pool = GlobalMaxPool1d()
        for c, k in zip(nums_channels, kernel_sizes):
            self.convs.append(nn.Conv1d(in_channels=in_dim,
                                        out_channels=c,
                                        kernel_size=k))

    def forward(self, inputs):
        inputs = inputs.permute(0,2,1)
        encoding = torch.cat([self.pool(F.relu(conv(inputs))).squeeze(-1) for conv in self.convs], dim=1)
        a = F.relu(self.convs[0](inputs)).squeeze(-1)
        b = F.relu(self.convs[1](inputs)).squeeze(-1)
        c = F.relu(self.convs[2](inputs)).squeeze(-1)
        # encoding = torch.cat([F.relu(conv(inputs)).squeeze(-1) for conv in self.convs], dim=1)
        # outputs = self.dropout(encoding)

        outputs = encoding
        return outputs




class Mybert(BertPreTrainedModel):
    def __init__(self, bert_config, args):
        super(Mybert, self).__init__(bert_config)
        self.bert = BertModel.from_pretrained(args.pretrained_model_name, config=bert_config)  # Load pretrained bert

        self.num_labels = bert_config.num_labels
        self.seq_multiatten = MultiAttn(768)
        self.seq_BiGRU = PackedGRU(bert_config.hidden_size, bert_config.hidden_size)
        self.cls_BiGRU = nn.GRU(bert_config.hidden_size, bert_config.hidden_size, num_layers=2, dropout=0,
                                bidirectional=True)
        # self.cls_fc_layer = FCLayer(bert_config.hidden_size, bert_config.hidden_size, args.dropout_rate)
        self.e1_fc_layer = FCLayer(bert_config.hidden_size * 2, bert_config.hidden_size, args.dropout_rate)
        self.e2_fc_layer = FCLayer(bert_config.hidden_size * 2, bert_config.hidden_size, args.dropout_rate)
        self.seq_fc_layer = FCLayer(bert_config.hidden_size * 4, bert_config.hidden_size, args.dropout_rate)
        self.label_classifier = FCLayer(bert_config.hidden_size * 4, bert_config.num_labels, args.dropout_rate,
                                        use_activation=False)
        # self.init_weights()

    @staticmethod
    def entity_average(hidden_output, e_mask):
        """
        Average the entity hidden state vectors (H_i ~ H_j)
        :param hidden_output: [batch_size, j-i+1, dim]
        :param e_mask: [batch_size, max_seq_len]
                e.g. e_mask[0] == [0, 0, 0, 1, 1, 1, 0, 0, ... 0]
        :return: [batch_size, dim]
        """
        e_mask_unsqueeze = e_mask.unsqueeze(1)  # [b, 1, j-i+1]
        length_tensor = (e_mask != 0).sum(dim=1).unsqueeze(1)  # [batch_size, 1]

        sum_vector = torch.bmm(e_mask_unsqueeze.float(), hidden_output).squeeze(1)  # [b, 1, j-i+1] * [b, j-i+1, dim] = [b, 1, dim] -> [b, dim]
        avg_vector = sum_vector.float() / length_tensor.float()  # broadcasting
        return avg_vector

    def forward(self, input_ids, attention_mask, token_type_ids, labels, e1_mask, e2_mask, length):
        attn_mask = padding_mask(input_ids)
        non_pad_mask = non_padding_mask(input_ids)
        outputs = self.bert(input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)  # sequence_output, pooled_output, (hidden_states), (attentions)
        sequence_output = outputs[0]
        pooled_output = outputs[1]  # [CLS]
        attensequence_output, attention_weight = self.seq_multiatten(sequence_output, attn_mask, non_pad_mask)
        grusequence_output = self.seq_BiGRU(attensequence_output, length)
        seq_1 = grusequence_output[:, 0, :]
        seq_2 = grusequence_output[:, -1, :]
        seq = torch.cat([seq_1, seq_2], dim=-1)
        seq = seq.squeeze(1)
        seq = self.seq_fc_layer(seq)
        # Average
        e1_h = self.entity_average(grusequence_output, e1_mask)
        e2_h = self.entity_average(grusequence_output, e2_mask)
        # Dropout -> tanh -> fc_layer
        # pooled_output, _ = self.cls_BiGRU(pooled_output)
        # pooled_output = self.cls_fc_layer(pooled_output)
        e1_h = self.e1_fc_layer(e1_h)
        e2_h = self.e2_fc_layer(e2_h)
        # Concat -> fc_layer
        concat_h = torch.cat([pooled_output, e1_h, e2_h, seq], dim=-1)
        logits = self.label_classifier(concat_h)

        return logits, attention_weight

class Mybert_CNN(BertPreTrainedModel):
    def __init__(self, bert_config, args):
        super(Mybert_CNN, self).__init__(bert_config)
        self.bert = BertModel.from_pretrained(args.pretrained_model_name, config=bert_config)  # Load pretrained bert

        self.num_labels = bert_config.num_labels
        self.seq_multiatten = MultiAttn(768)
        self.textCNN = textCNN(768, [3,4,5], [256,256,256])
        self.seq_BiGRU = PackedGRU(bert_config.hidden_size, bert_config.hidden_size)
        self.cls_fc_layer = FCLayer(bert_config.hidden_size, bert_config.hidden_size, args.dropout_rate)
        self.cnn_fc_layer = FCLayer(bert_config.hidden_size, bert_config.hidden_size, args.dropout_rate)
        self.e1_fc_layer = FCLayer(bert_config.hidden_size * 2, bert_config.hidden_size, args.dropout_rate)
        self.e2_fc_layer = FCLayer(bert_config.hidden_size * 2, bert_config.hidden_size, args.dropout_rate)
        self.seq_fc_layer = FCLayer(bert_config.hidden_size, bert_config.hidden_size, args.dropout_rate)
        self.label_classifier = FCLayer(bert_config.hidden_size * 4, bert_config.num_labels, args.dropout_rate,
                                        use_activation=False)
        # self.init_weights()

    @staticmethod
    def entity_average(hidden_output, e_mask):
        """
        Average the entity hidden state vectors (H_i ~ H_j)
        :param hidden_output: [batch_size, j-i+1, dim]
        :param e_mask: [batch_size, max_seq_len]
                e.g. e_mask[0] == [0, 0, 0, 1, 1, 1, 0, 0, ... 0]
        :return: [batch_size, dim]
        """
        e_mask_unsqueeze = e_mask.unsqueeze(1)  # [b, 1, j-i+1]
        length_tensor = (e_mask != 0).sum(dim=1).unsqueeze(1)  # [batch_size, 1]

        sum_vector = torch.bmm(e_mask_unsqueeze.float(), hidden_output).squeeze(1)  # [b, 1, j-i+1] * [b, j-i+1, dim] = [b, 1, dim] -> [b, dim]
        avg_vector = sum_vector.float() / length_tensor.float()  # broadcasting
        return avg_vector

    def forward(self, input_ids, attention_mask, token_type_ids, labels, e1_mask, e2_mask, length):
        attn_mask = padding_mask(input_ids)
        non_pad_mask = non_padding_mask(input_ids)
        outputs = self.bert(input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)  # sequence_output, pooled_output, (hidden_states), (attentions)
        sequence_output = outputs[0]
        pooled_output = outputs[1]  # [CLS]
        pooled_output = self.cls_fc_layer(pooled_output)
        attensequence_output, attention_weight = self.seq_multiatten(sequence_output, attn_mask, non_pad_mask)
        cnnsequence_output = self.textCNN(attensequence_output)
        cnnsequence_output= self.cnn_fc_layer(cnnsequence_output)
        grusequence_output = self.seq_BiGRU(attensequence_output, length)
        # seq_1 = grusequence_output[:, 0, :]
        # seq_2 = grusequence_output[:, -1, :]
        # seq = torch.cat([seq_1, seq_2], dim=-1)
        # seq = seq.squeeze(1)
        # seq = self.seq_fc_layer(seq)
        # Average
        e1_h = self.entity_average(grusequence_output, e1_mask)
        e2_h = self.entity_average(grusequence_output, e2_mask)
        # Dropout -> tanh -> fc_layer
        # pooled_output, _ = self.cls_BiGRU(pooled_output)
        # pooled_output = self.cls_fc_layer(pooled_output)

        # cnnsequence_output = self.seq_fc_layer(cnnsequence_output)
        e1_h = self.e1_fc_layer(e1_h)
        e2_h = self.e2_fc_layer(e2_h)
        # Concat -> fc_layer
        concat_h = torch.cat([cnnsequence_output, e1_h, e2_h, pooled_output], dim=-1)
        logits = self.label_classifier(concat_h)

        return logits, attention_weight

# class Mybert_startent(BertPreTrainedModel):
#     def __init__(self, bert_config, args):
#         super(Mybert_startent, self).__init__(bert_config)
#         self.bert = BertModel.from_pretrained(args.pretrained_model_name, config=bert_config)  # Load pretrained bert
#
#         self.num_labels = bert_config.num_labels
#         self.seq_multiatten = MultiAttn(768)
#         self.seq_BiGRU = PackedGRU(bert_config.hidden_size, bert_config.hidden_size)
#
#         self.cls_fc_layer = FCLayer(bert_config.hidden_size, bert_config.hidden_size, args.dropout_rate)
#         self.e1_fc_layer = FCLayer(bert_config.hidden_size * 2, bert_config.hidden_size, args.dropout_rate)
#         self.e2_fc_layer = FCLayer(bert_config.hidden_size * 2, bert_config.hidden_size, args.dropout_rate)
#         self.seq_fc_layer = FCLayer(bert_config.hidden_size * 4, bert_config.hidden_size, args.dropout_rate)
#         self.label_classifier = FCLayer(bert_config.hidden_size * 4, bert_config.num_labels, args.dropout_rate,
#                                         use_activation=False)
#         # self.init_weights()
#
#     @staticmethod
#     def entity_average(hidden_output, e_mask):
#         """
#         Average the entity hidden state vectors (H_i ~ H_j)
#         :param hidden_output: [batch_size, j-i+1, dim]
#         :param e_mask: [batch_size, max_seq_len]
#                 e.g. e_mask[0] == [0, 0, 0, 1, 1, 1, 0, 0, ... 0]
#         :return: [batch_size, dim]
#         """
#         e_mask_unsqueeze = e_mask.unsqueeze(1)  # [b, 1, j-i+1]
#         length_tensor = (e_mask != 0).sum(dim=1).unsqueeze(1)  # [batch_size, 1]
#
#         sum_vector = torch.bmm(e_mask_unsqueeze.float(), hidden_output).squeeze(1)  # [b, 1, j-i+1] * [b, j-i+1, dim] = [b, 1, dim] -> [b, dim]
#         avg_vector = sum_vector.float() / length_tensor.float()  # broadcasting
#         return avg_vector
#
#     def forward(self, input_ids, attention_mask, token_type_ids, labels, e1_mask, e2_mask, length):
#         attn_mask = padding_mask(input_ids)
#         non_pad_mask = non_padding_mask(input_ids)
#         outputs = self.bert(input_ids, attention_mask=attention_mask,
#                             token_type_ids=token_type_ids)  # sequence_output, pooled_output, (hidden_states), (attentions)
#         sequence_output = outputs[0]
#         pooled_output = outputs[1]  # [CLS]
#         attensequence_output, attention_weight = self.seq_multiatten(sequence_output, attn_mask, non_pad_mask)
#         grusequence_output = self.seq_BiGRU(attensequence_output, length)
#         seq_1 = grusequence_output[:, 0, :]
#         seq_2 = grusequence_output[:, -1, :]
#         seq = torch.cat([seq_1, seq_2], dim=-1)
#         seq = seq.squeeze(1)
#         seq = self.seq_fc_layer(seq)
#         # Average
#         e1_h = self.entity_average(grusequence_output, e1_mask)
#         e2_h = self.entity_average(grusequence_output, e2_mask)
#         e1_h = self.e1_fc_layer(e1_h)
#         e2_h = self.e2_fc_layer(e2_h)
#         # Dropout -> tanh -> fc_layer
#         # pooled_output, _ = self.cls_BiGRU(pooled_output)
#         pooled_output = self.cls_fc_layer(pooled_output)
#
#         # Concat -> fc_layer
#         concat_h = torch.cat([pooled_output, e1_h, e2_h, seq], dim=-1)
#         logits = self.label_classifier(concat_h)
#
#         return logits, attention_weight

class Mybert_startent(BertPreTrainedModel):
    def __init__(self, bert_config, args):
        super(Mybert_startent, self).__init__(bert_config)
        self.bert = BertModel.from_pretrained(args.pretrained_model_name, config=bert_config)  # Load pretrained bert

        self.num_labels = bert_config.num_labels
        self.seq_multiatten = MultiAttn(768)
        self.seq_BiGRU = PackedGRU(bert_config.hidden_size, bert_config.hidden_size)
        # self.cls_BiGRU = nn.GRU(bert_config.hidden_size, bert_config.hidden_size, num_layers=2, dropout=0,
        #                         bidirectional=True)
        self.cls_fc_layer = FCLayer(bert_config.hidden_size, bert_config.hidden_size, args.dropout_rate)
        self.ent_fc_layer = FCLayer(bert_config.hidden_size*4, bert_config.hidden_size, args.dropout_rate)
        self.seq_fc_layer = FCLayer(bert_config.hidden_size * 4, bert_config.hidden_size, args.dropout_rate)
        self.label_classifier = FCLayer(bert_config.hidden_size * 2, bert_config.num_labels, args.dropout_rate,
                                        use_activation=False)
        # self.init_weights()

    @staticmethod
    def entity_average(hidden_output, e_mask):
        """
        Average the entity hidden state vectors (H_i ~ H_j)
        :param hidden_output: [batch_size, j-i+1, dim]
        :param e_mask: [batch_size, max_seq_len]
                e.g. e_mask[0] == [0, 0, 0, 1, 1, 1, 0, 0, ... 0]
        :return: [batch_size, dim]
        """
        e_mask_unsqueeze = e_mask.unsqueeze(1)  # [b, 1, j-i+1]
        length_tensor = (e_mask != 0).sum(dim=1).unsqueeze(1)  # [batch_size, 1]

        sum_vector = torch.bmm(e_mask_unsqueeze.float(), hidden_output).squeeze(1)  # [b, 1, j-i+1] * [b, j-i+1, dim] = [b, 1, dim] -> [b, dim]
        avg_vector = sum_vector.float() / length_tensor.float()  # broadcasting
        return avg_vector

    def forward(self, input_ids, attention_mask, token_type_ids, labels, e1_mask, e2_mask, length):
        attn_mask = padding_mask(input_ids)
        non_pad_mask = non_padding_mask(input_ids)
        outputs = self.bert(input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)  # sequence_output, pooled_output, (hidden_states), (attentions)
        sequence_output = outputs[0]
        pooled_output = outputs[1]  # [CLS]
        attensequence_output, attention_weight = self.seq_multiatten(sequence_output, attn_mask, non_pad_mask)
        grusequence_output = self.seq_BiGRU(attensequence_output, length)
        seq_1 = grusequence_output[:, 0, :]
        seq_2 = grusequence_output[:, -1, :]
        seq = torch.cat([seq_1, seq_2], dim=-1)
        seq = seq.squeeze(1)
        seq = self.seq_fc_layer(seq)
        # Average
        e1_h = self.entity_average(grusequence_output, e1_mask)
        e2_h = self.entity_average(grusequence_output, e2_mask)
        entity = torch.cat([e1_h,e2_h], dim=-1)
        entity = self.ent_fc_layer(entity)
        # Dropout -> tanh -> fc_layer
        # pooled_output, _ = self.cls_BiGRU(pooled_output)
        pooled_output = self.cls_fc_layer(pooled_output)

        # Concat -> fc_layer
        concat_h = torch.cat([entity, seq], dim=-1)
        logits = self.label_classifier(concat_h)

        return logits

class Mybert_without_attention(BertPreTrainedModel):
    def __init__(self, bert_config, args):
        super(Mybert_without_attention, self).__init__(bert_config)
        self.bert = BertModel.from_pretrained(args.pretrained_model_name, config=bert_config)  # Load pretrained bert

        self.num_labels = bert_config.num_labels

        self.seq_BiGRU = PackedGRU(bert_config.hidden_size, bert_config.hidden_size)
        self.cls_BiGRU = nn.GRU(bert_config.hidden_size, bert_config.hidden_size, num_layers=2, dropout=0,
                                bidirectional=True)
        self.cls_fc_layer = FCLayer(bert_config.hidden_size, bert_config.hidden_size, args.dropout_rate)
        self.e1_fc_layer = FCLayer(bert_config.hidden_size * 2, bert_config.hidden_size, args.dropout_rate)
        self.e2_fc_layer = FCLayer(bert_config.hidden_size * 2, bert_config.hidden_size, args.dropout_rate)
        self.seq_fc_layer = FCLayer(bert_config.hidden_size * 4, bert_config.hidden_size, args.dropout_rate)
        self.label_classifier = FCLayer(bert_config.hidden_size * 4, bert_config.num_labels, args.dropout_rate,
                                        use_activation=False)
        # self.init_weights()

    @staticmethod
    def entity_average(hidden_output, e_mask):
        """
        Average the entity hidden state vectors (H_i ~ H_j)
        :param hidden_output: [batch_size, j-i+1, dim]
        :param e_mask: [batch_size, max_seq_len]
                e.g. e_mask[0] == [0, 0, 0, 1, 1, 1, 0, 0, ... 0]
        :return: [batch_size, dim]
        """
        e_mask_unsqueeze = e_mask.unsqueeze(1)  # [b, 1, j-i+1]
        length_tensor = (e_mask != 0).sum(dim=1).unsqueeze(1)  # [batch_size, 1]

        sum_vector = torch.bmm(e_mask_unsqueeze.float(), hidden_output).squeeze(1)  # [b, 1, j-i+1] * [b, j-i+1, dim] = [b, 1, dim] -> [b, dim]
        avg_vector = sum_vector.float() / length_tensor.float()  # broadcasting
        return avg_vector

    def forward(self, input_ids, attention_mask, token_type_ids, labels, e1_mask, e2_mask, length):
        attn_mask = padding_mask(input_ids)
        non_pad_mask = non_padding_mask(input_ids)
        outputs = self.bert(input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)  # sequence_output, pooled_output, (hidden_states), (attentions)
        sequence_output = outputs[0]
        pooled_output = outputs[1]  # [CLS]
        grusequence_output = self.seq_BiGRU(sequence_output, length)
        seq_1 = grusequence_output[:, 0, :]
        seq_2 = grusequence_output[:, -1, :]
        seq = torch.cat([seq_1, seq_2], dim=-1)
        seq = seq.squeeze(1)
        seq = self.seq_fc_layer(seq)
        # Average
        e1_h = self.entity_average(grusequence_output, e1_mask)
        e2_h = self.entity_average(grusequence_output, e2_mask)
        # Dropout -> tanh -> fc_layer
        # pooled_output, _ = self.cls_BiGRU(pooled_output)
        pooled_output = self.cls_fc_layer(pooled_output)
        e1_h = self.e1_fc_layer(e1_h)
        e2_h = self.e2_fc_layer(e2_h)
        # Concat -> fc_layer
        concat_h = torch.cat([pooled_output, e1_h, e2_h, seq], dim=-1)
        logits = self.label_classifier(concat_h)

        return logits

class Mybert_without_packedBiGRU(BertPreTrainedModel):
    def __init__(self, bert_config, args):
        super(Mybert_without_packedBiGRU, self).__init__(bert_config)
        self.bert = BertModel.from_pretrained(args.pretrained_model_name, config=bert_config)  # Load pretrained bert

        self.num_labels = bert_config.num_labels
        self.seq_multiatten = MultiAttn(768)
        self.cls_fc_layer = FCLayer(bert_config.hidden_size, bert_config.hidden_size, args.dropout_rate)
        self.e1_fc_layer = FCLayer(bert_config.hidden_size * 1, bert_config.hidden_size, args.dropout_rate)
        self.e2_fc_layer = FCLayer(bert_config.hidden_size * 1, bert_config.hidden_size, args.dropout_rate)
        self.seq_fc_layer = FCLayer(bert_config.hidden_size * 2, bert_config.hidden_size, args.dropout_rate)
        self.label_classifier = FCLayer(bert_config.hidden_size * 4, bert_config.num_labels, args.dropout_rate,
                                        use_activation=False)
        # self.init_weights()

    @staticmethod
    def entity_average(hidden_output, e_mask):
        """
        Average the entity hidden state vectors (H_i ~ H_j)
        :param hidden_output: [batch_size, j-i+1, dim]
        :param e_mask: [batch_size, max_seq_len]
                e.g. e_mask[0] == [0, 0, 0, 1, 1, 1, 0, 0, ... 0]
        :return: [batch_size, dim]
        """
        e_mask_unsqueeze = e_mask.unsqueeze(1)  # [b, 1, j-i+1]
        length_tensor = (e_mask != 0).sum(dim=1).unsqueeze(1)  # [batch_size, 1]

        sum_vector = torch.bmm(e_mask_unsqueeze.float(), hidden_output).squeeze(1)  # [b, 1, j-i+1] * [b, j-i+1, dim] = [b, 1, dim] -> [b, dim]
        avg_vector = sum_vector.float() / length_tensor.float()  # broadcasting
        return avg_vector

    def forward(self, input_ids, attention_mask, token_type_ids, labels, e1_mask, e2_mask, length):
        attn_mask = padding_mask(input_ids)
        non_pad_mask = non_padding_mask(input_ids)
        outputs = self.bert(input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)  # sequence_output, pooled_output, (hidden_states), (attentions)
        sequence_output = outputs[0]
        pooled_output = outputs[1]  # [CLS]
        attensequence_output, _ = self.seq_multiatten(sequence_output, attn_mask, non_pad_mask)

        seq_1 = attensequence_output[:, 0, :]
        seq_2 = attensequence_output[:, -1, :]
        seq = torch.cat([seq_1, seq_2], dim=-1)
        seq = seq.squeeze(1)
        seq = self.seq_fc_layer(seq)
        # Average
        e1_h = self.entity_average(attensequence_output, e1_mask)
        e2_h = self.entity_average(attensequence_output, e2_mask)
        # Dropout -> tanh -> fc_layer
        # pooled_output, _ = self.cls_BiGRU(pooled_output)
        pooled_output = self.cls_fc_layer(pooled_output)
        e1_h = self.e1_fc_layer(e1_h)
        e2_h = self.e2_fc_layer(e2_h)
        # Concat -> fc_layer
        concat_h = torch.cat([pooled_output, e1_h, e2_h, seq], dim=-1)
        logits = self.label_classifier(concat_h)

        return logits

class Mybert_without_entity_information(BertPreTrainedModel):
    def __init__(self, bert_config, args):
        super(Mybert_without_entity_information, self).__init__(bert_config)
        self.bert = BertModel.from_pretrained(args.pretrained_model_name, config=bert_config)  # Load pretrained bert

        self.num_labels = bert_config.num_labels
        self.seq_multiatten = MultiAttn(768)
        self.seq_BiGRU = PackedGRU(bert_config.hidden_size, bert_config.hidden_size)
        self.cls_BiGRU = nn.GRU(bert_config.hidden_size, bert_config.hidden_size, num_layers=2, dropout=0,
                                bidirectional=True)
        self.cls_fc_layer = FCLayer(bert_config.hidden_size, bert_config.hidden_size, args.dropout_rate)
        self.seq_fc_layer = FCLayer(bert_config.hidden_size * 4, bert_config.hidden_size, args.dropout_rate)
        self.label_classifier = FCLayer(bert_config.hidden_size * 2, bert_config.num_labels, args.dropout_rate,
                                        use_activation=False)
        # self.init_weights()


    def forward(self, input_ids, attention_mask, token_type_ids, labels, e1_mask, e2_mask, length):
        attn_mask = padding_mask(input_ids)
        non_pad_mask = non_padding_mask(input_ids)
        outputs = self.bert(input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)  # sequence_output, pooled_output, (hidden_states), (attentions)
        sequence_output = outputs[0]
        pooled_output = outputs[1]  # [CLS]
        attensequence_output, _ = self.seq_multiatten(sequence_output, attn_mask, non_pad_mask)
        grusequence_output = self.seq_BiGRU(attensequence_output, length)
        seq_1 = grusequence_output[:, 0, :]
        seq_2 = grusequence_output[:, -1, :]
        seq = torch.cat([seq_1, seq_2], dim=-1)
        seq = seq.squeeze(1)
        seq = self.seq_fc_layer(seq)

        concat_h = torch.cat([pooled_output, seq], dim=-1)
        logits = self.label_classifier(concat_h)

        return logits

class Mybert_without_sentence_information(BertPreTrainedModel):
    def __init__(self, bert_config, args):
        super(Mybert_without_sentence_information, self).__init__(bert_config)
        self.bert = BertModel.from_pretrained(args.pretrained_model_name, config=bert_config)  # Load pretrained bert

        self.num_labels = bert_config.num_labels
        self.seq_multiatten = MultiAttn(768)
        self.seq_BiGRU = PackedGRU(bert_config.hidden_size, bert_config.hidden_size)

        self.e1_fc_layer = FCLayer(bert_config.hidden_size * 2, bert_config.hidden_size, args.dropout_rate)
        self.e2_fc_layer = FCLayer(bert_config.hidden_size * 2, bert_config.hidden_size, args.dropout_rate)
        self.label_classifier = FCLayer(bert_config.hidden_size * 2, bert_config.num_labels, args.dropout_rate,
                                        use_activation=False)
        # self.init_weights()

    @staticmethod
    def entity_average(hidden_output, e_mask):
        """
        Average the entity hidden state vectors (H_i ~ H_j)
        :param hidden_output: [batch_size, j-i+1, dim]
        :param e_mask: [batch_size, max_seq_len]
                e.g. e_mask[0] == [0, 0, 0, 1, 1, 1, 0, 0, ... 0]
        :return: [batch_size, dim]
        """
        e_mask_unsqueeze = e_mask.unsqueeze(1)  # [b, 1, j-i+1]
        length_tensor = (e_mask != 0).sum(dim=1).unsqueeze(1)  # [batch_size, 1]

        sum_vector = torch.bmm(e_mask_unsqueeze.float(), hidden_output).squeeze(
            1)  # [b, 1, j-i+1] * [b, j-i+1, dim] = [b, 1, dim] -> [b, dim]
        avg_vector = sum_vector.float() / length_tensor.float()  # broadcasting
        return avg_vector

    def forward(self, input_ids, attention_mask, token_type_ids, labels, e1_mask, e2_mask, length):
        attn_mask = padding_mask(input_ids)
        non_pad_mask = non_padding_mask(input_ids)
        outputs = self.bert(input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)  # sequence_output, pooled_output, (hidden_states), (attentions)
        sequence_output = outputs[0]

        attensequence_output, _ = self.seq_multiatten(sequence_output, attn_mask, non_pad_mask)
        grusequence_output = self.seq_BiGRU(attensequence_output, length)
        # Average
        e1_h = self.entity_average(grusequence_output, e1_mask)
        e2_h = self.entity_average(grusequence_output, e2_mask)
        # Dropout -> tanh -> fc_layer
        # pooled_output, _ = self.cls_BiGRU(pooled_output)

        e1_h = self.e1_fc_layer(e1_h)
        e2_h = self.e2_fc_layer(e2_h)
        # Concat -> fc_layer
        concat_h = torch.cat([e1_h, e2_h], dim=-1)
        logits = self.label_classifier(concat_h)

        return logits

class MyRoberta(BertPreTrainedModel):
    def __init__(self, bert_config, args):
        super(MyRoberta, self).__init__(bert_config)
        self.bert = RobertaModel.from_pretrained(args.pretrained_model_name, config=bert_config)  # Load pretrained bert

        self.num_labels = bert_config.num_labels

        self.seq_multiatten = MultiAttn(bert_config.hidden_size)
        self.seq_BiGRU = PackedGRU(bert_config.hidden_size, bert_config.hidden_size, bidirectional=True)

        self.cls_fc_layer = FCLayer(bert_config.hidden_size, bert_config.hidden_size, args.dropout_rate)
        self.e1_fc_layer = FCLayer(bert_config.hidden_size * 2, bert_config.hidden_size, args.dropout_rate)
        self.e2_fc_layer = FCLayer(bert_config.hidden_size * 2, bert_config.hidden_size, args.dropout_rate)
        self.seq_fc_layer = FCLayer(bert_config.hidden_size * 4, bert_config.hidden_size, args.dropout_rate)
        self.label_classifier = FCLayer(bert_config.hidden_size * 4, bert_config.num_labels, args.dropout_rate,
                                        use_activation=False)

    @staticmethod
    def entity_average(hidden_output, e_mask):
        """
        Average the entity hidden state vectors (H_i ~ H_j)
        :param hidden_output: [batch_size, j-i+1, dim]
        :param e_mask: [batch_size, max_seq_len]
                e.g. e_mask[0] == [0, 0, 0, 1, 1, 1, 0, 0, ... 0]
        :return: [batch_size, dim]
        """
        e_mask_unsqueeze = e_mask.unsqueeze(1)  # [b, 1, j-i+1]
        length_tensor = (e_mask != 0).sum(dim=1).unsqueeze(1)  # [batch_size, 1]

        sum_vector = torch.bmm(e_mask_unsqueeze.float(), hidden_output).squeeze(1)  # [b, 1, j-i+1] * [b, j-i+1, dim] = [b, 1, dim] -> [b, dim]
        avg_vector = sum_vector.float() / length_tensor.float()  # broadcasting
        return avg_vector

    def forward(self, input_ids, attention_mask, token_type_ids, labels, e1_mask, e2_mask,length):
        attn_mask = padding_mask(input_ids)
        non_pad_mask = non_padding_mask(input_ids)
        outputs = self.bert(input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)  # sequence_output, pooled_output, (hidden_states), (attentions)
        sequence_output = outputs[0]
        pooled_output = outputs[1]  # [CLS]
        attensequence_output, _ = self.seq_multiatten(sequence_output, attn_mask, non_pad_mask)
        grusequence_output = self.seq_BiGRU(attensequence_output, length)
        seq_1 = grusequence_output[:, 0, :]
        seq_2 = grusequence_output[:, -1, :]
        seq = torch.cat([seq_1, seq_2], dim=-1)
        seq = seq.squeeze(1)
        seq = self.seq_fc_layer(seq)
        # Average
        e1_h = self.entity_average(grusequence_output, e1_mask)
        e2_h = self.entity_average(grusequence_output, e2_mask)
        # Dropout -> tanh -> fc_layer
        # pooled_output, _ = self.cls_BiGRU(pooled_output)
        pooled_output = self.cls_fc_layer(pooled_output)
        e1_h = self.e1_fc_layer(e1_h)
        e2_h = self.e2_fc_layer(e2_h)
        # Concat -> fc_layer
        concat_h = torch.cat([pooled_output, e1_h, e2_h, seq], dim=-1)
        logits = self.label_classifier(concat_h)

        return logits

