"""
@uthor: Prakhar
"""
import os
import argparse
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import numpy as np
import warnings
import csv
from tqdm import tqdm

warnings.filterwarnings('ignore')


def choose_from_top_k_top_n(probs, k=50, p=0.8):
    ind = np.argpartition(probs, -k)[-k:]
    top_prob = probs[ind]
    top_prob = {i: top_prob[idx] for idx, i in enumerate(ind)}
    sorted_top_prob = {k: v for k, v in sorted(top_prob.items(), key=lambda item: item[1], reverse=True)}

    t = 0
    f = []
    pr = []
    for k, v in sorted_top_prob.items():
        t += v
        f.append(k)
        pr.append(v)
        if t >= p:
            break
    top_prob = pr / np.sum(pr)
    token_id = np.random.choice(f, 1, p=top_prob)

    return int(token_id)


def generate(tokenizer, model, sentence, label, DEVICE):
    label_gen = open(label + "_g.tsv", 'w+', newline='', encoding='utf-8')
    writer = csv.writer(label_gen, delimiter='\t')
    device = DEVICE
    with torch.no_grad():
        for idx in tqdm(range(sentence)):
            finished = False
            cur_ids = torch.tensor(tokenizer.encode(label)).unsqueeze(0).to(device)
            for i in range(100):
                outputs = model(cur_ids, labels=cur_ids)
                loss, logits = outputs[:2]

                softmax_logits = torch.softmax(logits[0, -1], dim=0)

                if i < 5:
                    n = 10
                else:
                    n = 5

                next_token_id = choose_from_top_k_top_n(softmax_logits.to('cpu').numpy())  # top-k-top-n sampling
                cur_ids = torch.cat([cur_ids, torch.ones((1, 1)).long().to(device) * next_token_id], dim=1)
                # cur_ids = torch.ones((1, 1)).long().to(device) * next_token_id

                if next_token_id in tokenizer.encode('<|endoftext|>'):
                    finished = True
                    break

            if finished:
                output_list = list(cur_ids.squeeze().to('cpu').numpy())
                output_text = tokenizer.decode(output_list)
                # try:
                #     writer.writerow([label, output_text[len(label) + 2:].replace('<|endoftext|>', '')])
                # except UnicodeEncodeError:
                #     print(r"gbk' codec can't encode character '\xae' in position 20: illegal multibyte sequence")
                #     continue

                print(output_text.replace('<|endoftext|>',''))
            else:
                output_list = list(cur_ids.squeeze().to('cpu').numpy())
                output_text = tokenizer.decode(output_list)
                # try:
                #     writer.writerow([label, output_text[len(label) + 2:].replace('<|endoftext|>', '')])
                # except UnicodeEncodeError:
                #     print(r"gbk' codec can't encode character '\xae' in position 20: illegal multibyte sequence")
                #     continue
                print (output_text.replace('<|endoftext|>',''))


def load_models(model_name):
    """
    Summary:
        Loading the trained model
    """
    print('Loading Trained GPT-2 Model')
    tokenizer = GPT2Tokenizer.from_pretrained('healx/gpt-2-pubmed-large')
    model = GPT2LMHeadModel.from_pretrained('healx/gpt-2-pubmed-large')
    model_path = model_name
    model.load_state_dict(torch.load(model_path))
    return tokenizer, model


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Arguments for inferencing Text Augmentation model')

    parser.add_argument('--model_name', default='effect_model.pt.pt', type=str, action='store',
                        help='Name of the model file')
    parser.add_argument('--sentences', type=int, default=5, action='store', help='Number of sentences in outputs')
    parser.add_argument('--label', type=str, default='effect', action='store', help='Label for which to produce text')
    args = parser.parse_args()

    SENTENCES = args.sentences
    MODEL_NAME = args.model_name
    LABEL = args.label
    # print(LABEL)
    # exit()

    DEVICE = 'cpu'
    if torch.cuda.is_available():
        DEVICE = 'cuda'

    TOKENIZER, MODEL = load_models(MODEL_NAME)
    MODEL = MODEL.to(DEVICE)

    generate(TOKENIZER, MODEL, SENTENCES, LABEL, DEVICE)
