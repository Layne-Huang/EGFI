import os
import argparse
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import numpy as np
import warnings
import csv
from tqdm import tqdm
import math
from tqdm import tqdm

test_data = open('data/test_c.tsv')
reader = csv.reader(test_data, delimiter='\t')
negative = []
advise = []
effect = []
mechanism = []
int = []
label = ['negative', 'advise', 'effect', 'mechanism', 'int']

model1_name = 'overallmodel.pt'
tokenizer = GPT2Tokenizer.from_pretrained('healx/gpt-2-pubmed-large')
model1 = GPT2LMHeadModel.from_pretrained('healx/gpt-2-pubmed-large')
model1_path = model1_name
model1.load_state_dict(torch.load(model1_path))

model2_name = 'advise_model.pt.pt'
model2 = GPT2LMHeadModel.from_pretrained('healx/gpt-2-pubmed-large')
model2_path = model2_name
model2.load_state_dict(torch.load(model2_path))

def get_score(sentence,model):
    tokens = tokenizer.tokenize(sentence)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_tensor = torch.tensor([input_ids])
    with torch.no_grad():
        outputs = model(input_tensor,labels=input_tensor)
        loss, logits = outputs[0:2]
        return math.exp(loss.item())

for x in reader:
    if x[0]=='negative':
        negative.append(x)
    if x[0]=='advise':
        advise.append(x)
    if x[0]=='effect':
        effect.append(x)
    if x[0]=='mechanism':
        mechanism.append(x)
    if x[0]=='int':
        int.append(x)

loss_all = 0
loss_effect = 0

for x in tqdm(int):
    l_all = get_score(x[1],model1)
    loss_all += l_all
    l_e = get_score(x[1], model2)
    loss_effect += l_e

print(loss_all)
print(loss_effect)



