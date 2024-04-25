import sys
from collections import Counter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from transformers import BertPreTrainedModel, BertTokenizer, BertConfig, BertModel, AutoConfig
from functools import partial
from transformers import AdamW, get_linear_schedule_with_warmup

sys.path.append('..')
SEED = 308
torch.manual_seed(SEED)
origin_train_data_path = '.\\public_data\\Train.csv'
origin_test_data_path = '.\\public_data\\Test.csv'
pred_path = '.\\dataset\\submission.txt'
LABEL_LIST = ['positive', 'neutral', 'negative']


def get_public_data(is_train=True):
    return pd.read_csv(origin_train_data_path if is_train else origin_test_data_path)


def train_data_distribution(_data_set: pd.DataFrame):
    labels = _data_set['labels']
    b = Counter(labels)
    print(b)


class MyDataset(Dataset):
    def __init__(self, tokenizer, max_len, _dataframe, mode='train'):
        super(MyDataset, self).__init__()
        self.data = _dataframe
        self.texts = self.data['text'].tolist()
        if mode == 'train':
            self.labels = self.data['labels'].tolist()
        else:
            self.labels = ['positive'] * len(self.texts)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, index):
        text = str(self.texts[index])
        label = torch.zeros(3)
        label[LABEL_LIST.index(self.labels[index])] = 1
        encoding = self.tokenizer.encode_plus(text,
                                              padding='max_length',
                                              truncation=True,
                                              add_special_tokens=True,
                                              max_length=self.max_len,
                                              return_token_type_ids=True,
                                              return_attention_mask=True,
                                              return_tensors='pt', )

        sample = {
            'texts': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': label
        }
        return sample

    def __len__(self):
        return len(self.texts)


def create_dataloader(dataset, batch_size, mode='train'):
    shuffle = True if mode == 'train' else False

    if mode == 'train':
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    else:
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return data_loader


def save_pred(pred: list):
    with open(pred_path, 'w') as ans_f:
        for ans in tqdm(pred):
            ans_f.write(LABEL_LIST[ans] + "\n")


if __name__ == '__main__':
    data_set = get_public_data()
