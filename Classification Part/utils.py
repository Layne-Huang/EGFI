"""Basic or helper implementation."""

import glob
import os

import torch
from torch.nn import functional
from torch.utils.data import TensorDataset, ConcatDataset, Subset


def convert_to_one_hot(indices, num_classes):
    """
    Args:
        indices (tensor): A vector containing indices,
            whose size is (batch_size,).
        num_classes (tensor): The number of classes, which would be
            the second dimension of the resulting one-hot matrix.

    Returns:
        result: The one-hot matrix of size (batch_size, num_classes).
    """

    batch_size = indices.size(0)
    indices = indices.unsqueeze(1)
    one_hot = indices.new_zeros(batch_size, num_classes).scatter_(1, indices, 1)
    return one_hot


def masked_softmax(logits, mask=None):
    eps = 1e-20
    probs = functional.softmax(logits, dim=1)
    if mask is not None:
        mask = mask.float()
        probs = probs * mask + eps
        probs = probs / probs.sum(1, keepdim=True)
    return probs


def greedy_select(logits, mask=None):
    probs = masked_softmax(logits=logits, mask=mask)
    one_hot = convert_to_one_hot(indices=probs.max(1)[1],
                                 num_classes=logits.size(1))
    return one_hot


def st_gumbel_softmax(logits, temperature=1.0, mask=None):
    """
    Return the result of Straight-Through Gumbel-Softmax Estimation.
    It approximates the discrete sampling via Gumbel-Softmax trick
    and applies the biased ST estimator.
    In the forward propagation, it emits the discrete one-hot result,
    and in the backward propagation it approximates the categorical
    distribution via smooth Gumbel-Softmax distribution.

    Args:
        logits (tensor): A un-normalized probability values,
            which has the size (batch_size, num_classes)
        temperature (float): A temperature parameter. The higher
            the value is, the smoother the distribution is.
        mask (tensor, optional): If given, it masks the softmax
            so that indices of '0' mask values are not selected.
            The size is (batch_size, num_classes).

    Returns:
        y: The sampled output, which has the property explained above.
    """

    eps = 1e-20
    u = logits.data.new(*logits.size()).uniform_()
    gumbel_noise = -torch.log(-torch.log(u + eps) + eps)
    y = logits + gumbel_noise
    y = masked_softmax(logits=y / temperature, mask=mask)
    y_argmax = y.max(1)[1]
    y_hard = convert_to_one_hot(indices=y_argmax, num_classes=y.size(1)).float()
    y = (y_hard - y).detach() + y
    return y


def sequence_mask(length, max_length=None):
    if max_length is None:
        max_length = length.max()
    batch_size = length.size(0)
    seq_range = torch.arange(0, max_length).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_length)
    seq_range_expand = seq_range_expand.to(length)
    seq_length_expand = length.unsqueeze(1).expand_as(seq_range_expand)
    return seq_range_expand < seq_length_expand


def reverse_padded_sequence(inputs, lengths, batch_first=True):
    """Reverses sequences according to their lengths.
    Inputs should have size ``T x B x *`` if ``batch_first`` is False, or
    ``B x T x *`` if True. T is the length of the longest sequence (or larger),
    B is the batch size, and * is any number of dimensions (including 0).
    Arguments:
        inputs (tensor): padded batch of variable length sequences.
        lengths (list[int]): list of sequence lengths
        batch_first (bool, optional): if True, inputs should be B x T x *.
    Returns:
        A tensor with the same size as inputs, but with each sequence
        reversed according to its length.
    """

    if not batch_first:
        inputs = inputs.transpose(0, 1)
    if inputs.size(0) != len(lengths):
        raise ValueError('inputs incompatible with lengths.')
    reversed_indices = [list(range(inputs.size(1)))
                        for _ in range(inputs.size(0))]
    for i, length in enumerate(lengths):
        if length > 0:
            reversed_indices[i][:length] = reversed_indices[i][length - 1::-1]
    reversed_indices = (torch.LongTensor(reversed_indices).unsqueeze(2)
                        .expand_as(inputs))
    reversed_indices = reversed_indices.to(inputs.device)
    reversed_inputs = torch.gather(inputs, 1, reversed_indices)
    if not batch_first:
        reversed_inputs = reversed_inputs.transpose(0, 1)
    return reversed_inputs


def non_padding_mask(seq):
    return seq.ne(0).type(torch.float).unsqueeze(-1)


def value_mask(seq, value):
    return seq.eq(value).type(torch.float).unsqueeze(-1)


def padding_mask(seq):
    len_q = seq.size(1)
    pad_mask = seq.eq(0).unsqueeze(1).expand(-1, len_q, -1)  # shape [B, L_q, L_k]
    return pad_mask


def best_model_path(model_dir, model_name, i = 0):
    paths = glob.glob(os.path.join(model_dir, model_name, model_name + '*'))
    paths.sort(reverse=True)
    return paths[i]

def cattuple(a,b):
    for i,x in enumerate(a):
        # a[i] = a[i]
        # b[i] = b[i]
        # print(a[i])
        # print(b[i])
        # exit()
        a[i] = torch.cat((x,b[i]),0)
    return a

def skip_range(a,b):
    indices = []
    for x in a:
        if x not in b:
            indices.append(x)

    return indices

def get_k_fold_data(k, i, data):
    # 返回第i折交叉验证时所需要的训练和验证数据
    assert k > 1
    fold_size = len(data) // k
    data_train, data_valid = None, None
    # for j in range(k):
    #     idx = slice(j * fold_size, (j + 1) * fold_size)
    #     data_part = Subset(data,list(range(1,1000)))
    #     if j == i:
    #         data_valid = data_part
    #     elif data_train is None:
    #         data_train = data_part
    #     else:
    #         data_train = data_train.__add__(data_part)
    #         # data_train = cattuple(data_train, data_part)
    indices_all = list(range(0,len(data)))
    if i!=k-1:
        indices = list(range(i * fold_size, (i + 1) * fold_size))
    else:
        indices = list(range(i * fold_size, len(data)))

    indices_train = skip_range(indices_all,indices)
    data_train = Subset(data,indices_train)
    data_valid = Subset(data, indices)

    return data_train, data_valid

def split_data(r, data):
    valid_size = ()