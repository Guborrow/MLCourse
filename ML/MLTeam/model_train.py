from transformers import RobertaTokenizer, RobertaModel, AutoConfig, BertTokenizer, BertModel
from functools import partial
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader

from torchsummary import summary
import torch
import torch.nn as nn
from torch.nn import MaxPool1d
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os
import sys
import time
from countermeasure_training import FGM, PGD
from load_data import MyDataset, create_dataloader, get_public_data, save_pred
from tqdm import tqdm

sys.path.append('..')

PRETRAINED_MODEL_TYPE = 0

PRETRAINED_MODEL_NAME = ['roberta-base', 'roberta-large']
PRETRAINED_MODEL_PATH = os.path.join(os.path.abspath('.'), "pretrainedModel",
                                     PRETRAINED_MODEL_NAME[PRETRAINED_MODEL_TYPE])
# 加载模型
config = AutoConfig.from_pretrained(PRETRAINED_MODEL_PATH)
config.update({"output_hidden_states": True})
tokenizer = RobertaTokenizer.from_pretrained(PRETRAINED_MODEL_PATH)
base_model = RobertaModel.from_pretrained(PRETRAINED_MODEL_PATH, config=config)
# mast use cuda
# USE_CUDA = torch.cuda.is_available()

# 训练参数
MAX_LEN = 128
EPOCHS = 2
BATCH_SIZE = 16
weight_decay = 0.0
warmup_proportion = 0.1
lr = 1e-5

warm_up_ratio = 0.000
train_dataset = MyDataset(tokenizer=tokenizer, _dataframe=get_public_data(), max_len=MAX_LEN)
train_loader = create_dataloader(train_dataset, batch_size=BATCH_SIZE)
public_test_dataset = MyDataset(tokenizer=tokenizer, _dataframe=get_public_data(False), max_len=MAX_LEN, mode='test')
public_test_loader = create_dataloader(public_test_dataset, batch_size=BATCH_SIZE, mode='test')
total_steps = len(train_loader) * EPOCHS
criterion = nn.BCEWithLogitsLoss().cuda()


def init_params(module_lst):
    for module in module_lst:
        for param in module.parameters():
            if param.dim() > 1:
                torch.nn.init.xavier_uniform_(param)
    return


class SentencePositiveModel(nn.Module):
    def __init__(self, n_classes=3):
        super(SentencePositiveModel, self).__init__()
        self.base = base_model
        dim = 1024 if 'large' in PRETRAINED_MODEL_NAME[PRETRAINED_MODEL_TYPE] else 768
        dim *= 2
        self.attention = nn.Sequential(
            nn.Linear(dim, 512),
            nn.Tanh(),
            nn.Linear(512, 1),
            nn.Softmax(dim=1)
        )

        self.out = nn.Sequential(
            nn.Linear(dim, n_classes)
        )

        init_params([self.attention, self.out])

    def forward(self, input_ids, attention_mask):
        bert_output = self.base(input_ids=input_ids,
                                attention_mask=attention_mask)
        # 得到最后两层并直接拼接效果不好

        max_pool = MaxPool1d(1)
        last_layer_hidden_states1 = max_pool(bert_output.hidden_states[-1])
        last_layer_hidden_states2 = max_pool(bert_output.hidden_states[-2])
        last_layer_hidden_states = torch.cat([last_layer_hidden_states1, last_layer_hidden_states2], dim=-1)
        # last_layer_hidden_states = bert_output.hidden_states[-1]
        # 得到卷积注意力层的权重
        weights = self.attention(last_layer_hidden_states)
        context_vector = torch.sum(weights * last_layer_hidden_states, dim=1)
        out = self.out(context_vector)
        return out


def build_model():
    _model = SentencePositiveModel(n_classes=3)

    _model.cuda()

    if torch.cuda.device_count() > 1:
        _model = nn.DataParallel(_model)
    _optimizer = AdamW(_model.parameters(), lr=lr, weight_decay=weight_decay)  # correct_bias=False,
    _scheduler = get_linear_schedule_with_warmup(
        _optimizer,
        num_warmup_steps=warm_up_ratio * total_steps,
        num_training_steps=total_steps
    )
    return _model, _optimizer, _scheduler


def do_train_with_fgm():
    model.train()
    global_step = 0
    tic_train = time.time()
    log_steps = 100
    fgm = FGM(model)
    for epoch in range(EPOCHS):
        losses = []
        for step, sample in enumerate(train_loader):
            loss = get_loss(model, sample)

            losses.append(loss.item())
            loss.backward()

            attack = True
            # 对抗训练
            if attack:
                fgm.attack()
                loss_sum = get_loss(model, sample)
                loss_sum.backward()
                fgm.restore()

            # 优化器参数修改
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            global_step += 1

            if global_step % log_steps == 0:
                print("global step %d, epoch: %d, batch: %d, loss: %.5f, speed: %.2f step/s, lr: %.10f"
                      % (global_step, epoch, step, np.mean(losses), global_step / (time.time() - tic_train),
                         float(scheduler.get_last_lr()[0])))


def get_loss(input_model: nn.Module, sample):
    input_ids = sample["input_ids"].cuda()
    attention_mask = sample["attention_mask"].cuda()
    outputs = input_model(input_ids=input_ids, attention_mask=attention_mask)
    loss = criterion(outputs, sample['label'].cuda())
    return loss


def predict(test_loader):
    test_pred = []
    model.eval()
    model.cuda()
    for batch in tqdm(test_loader, desc='Pred'):
        b_input_ids = batch['input_ids'].cuda()

        attention_mask = batch["attention_mask"].cuda()
        with torch.no_grad():
            ans = model(input_ids=b_input_ids, attention_mask=attention_mask)
            ans = ans.cpu()
            ans = ans.numpy()
            ans = np.argmax(ans, axis=1)
            for _ in ans:
                test_pred.append(_)
    return test_pred


if __name__ == '__main__':
    model, optimizer, scheduler = build_model()
    do_train_with_fgm()
    print('Train Done')
    torch.save(model, f'.\\model\\fgm_{PRETRAINED_MODEL_NAME[PRETRAINED_MODEL_TYPE]}_maxpool_2layers.model')
    # model = torch.load(f'.\\model\\fgm_{PRETRAINED_MODEL_NAME[PRETRAINED_MODEL_TYPE]}.model')
    pred = predict(test_loader=public_test_loader)
    save_pred(pred)
