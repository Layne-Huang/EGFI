import os
import json
from tqdm import tqdm

files=['train.json','valid.json', 'test.json']

def merge_JsonFiles(filename):
    result = list()
    for f1 in filename:
        with open(f1, 'r') as infile:
            result.extend(json.load(infile))

    with open('full_dataset.json', 'w') as output_file:
        json.dump(result, output_file)

merge_JsonFiles(files)



