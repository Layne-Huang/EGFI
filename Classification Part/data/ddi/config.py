import logging
import os

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s %(message)s')

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))

SAVE_DIR = 'checkpoint/BioBert/CNN/drugmask/epoch'
# SAVE_DIR = 'checkpoint/BioBert'
RESULT_DIR = 'result'
OUTPUT_DIR = 'output'
DATA_SET = 'ddi_e-5'
DATA_SET = 'lossweight'

BAG_MODE = False
BERT_MODE = 1# 0代表Bert，1代表BioBert，2代表RioBerta
LOSS_WEIGHT = None

EMBEDDING_FINE_TUNE = True
BIDIRECTIONAL = True

# 这里设置的MAX_LENGTH?
MAX_LENGTH = 300
TAG_DIM = 50 # 标签的大小
HIDDEN_DIM = 250

DROP_PROB = 0.5
L2_REG = 0

max_steps = -1
LEARNING_RATE = 1
BATCH_SIZE = 8
MAX_EPOCHS = 20

CLASS_NUM = 15

adam_epsilon = 1e-8

max_steps = -1

gradient_accumulation_steps = 1

pretrained_model_name = 'monologg/biobert_v1.1_pubmed'
# pretrained_model_name = 'allenai/biomed_roberta_base'
# pretrained_model_name = 'minhpqn/bio_roberta-base_pubmed'
# pretrained_model_name ="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
# pretrained_model_name = "allenai/scibert_scivocab_uncased"
task = "ddi"

dropout_rate = 0.1
do_lower_case = True


def log():
    logging.info('Loading config of {}'.format(ROOT_DIR))

    logging.info('BAG_MODE {}'.format('✔' if BAG_MODE else '×'))
    logging.info('LOSS_WEIGHT: {}'.format(LOSS_WEIGHT))
    logging.info('EMBEDDING_FINE_TUNE {}'.format('✔' if EMBEDDING_FINE_TUNE else '×'))
    logging.info('BIDIRECTIONAL {}'.format('✔' if BIDIRECTIONAL else '×'))

    logging.info('MAX_LENGTH: {}'.format(MAX_LENGTH))
    logging.info('TAG_DIM: {}'.format(TAG_DIM))
    logging.info('HIDDEN_DIM: {}'.format(HIDDEN_DIM))

    logging.info('DROP_PROB: {}'.format(DROP_PROB))
    logging.info('L2_REG: {}'.format(L2_REG))

    logging.info('LEARNING_RATE: {}'.format(LEARNING_RATE))
    logging.info('BATCH_SIZE: {}'.format(BATCH_SIZE))
    logging.info('MAX_EPOCHS: {}'.format(MAX_EPOCHS))
