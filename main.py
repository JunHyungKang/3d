import h5py # .h5 파일을 읽기 위한 패키지
import random
import pandas as pd
import numpy as np
import os

import torch
from torch.utils.data import DataLoader
import tensorboard

from models.basemodel import BaseModel
import models.c3d
from train import train
from data import CustomDataset
from test import predict

import warnings
warnings.filterwarnings(action='ignore')

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

CFG = {
    'EPOCHS': 100,
    'LEARNING_RATE': 1e-3,
    'BATCH_SIZE': 512,
    'SEED': 77,
    'tr_csv': './data/train.csv',
    'tr_h5': './data/train.h5',
    'sample_csv': './data/sample_submission.csv',
    'test_h5': './data/test.h5',
    'save_path': './weights',
    'submit_file': 'submit_6.csv',
    'model': 'c3d'
}


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


seed_everything(CFG['SEED'])  # Seed 고정

# TODO: 5 split + 앙상블 코드 추가
train_df = pd.read_csv('./data/split_train.csv')
val_df = pd.read_csv('./data/split_val.csv')

if CFG['model'] == 'basemodel':
    model = BaseModel()
elif CFG['model'] == 'c3d':
    from models.c3d import get_fine_tuning_parameters
    model = models.c3d.get_model(
        num_classes=10,
        sample_size=16,
        sample_duration=16)

model.eval()

optimizer = torch.optim.Adam(params=model.parameters(), lr=CFG["LEARNING_RATE"])
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96, last_epoch=-1)

# optimizer = torch.optim.SGD(model.parameters(), lr=CFG["LEARNING_RATE"]*2, momentum=0.9)
# scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer=optimizer, base_lr=CFG["LEARNING_RATE"]/50,
#                                               max_lr=CFG["LEARNING_RATE"]*2, step_size_up=10,
#                                               step_size_down=None, mode='exp_range',
#                                               gamma=0.995, scale_fn=None, scale_mode='cycle',
#                                               cycle_momentum=True, base_momentum=0.8, max_momentum=0.9,
#                                               last_epoch=- 1, verbose=False)

#
train_dataset = CustomDataset(train_df['ID'].values, train_df['label'].values, augment=True, task='trainval')
train_loader = DataLoader(train_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=True)
val_dataset = CustomDataset(val_df['ID'].values, val_df['label'].values, augment=True, task='trainval')
val_loader = DataLoader(val_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False)

# TODO: 학습 비교 가능하도록 tensorboard 나 wandb 추가 (가능하면 wandb를 사용해보자!)
train(model, optimizer, train_loader, val_loader, scheduler, device, CFG)

test_df = pd.read_csv(CFG['sample_csv'])
test_points = h5py.File(CFG['test_h5'], 'r')
test_dataset = CustomDataset(test_df['ID'].values, None, augment=False, task='test')
test_loader = DataLoader(test_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False)

checkpoint = torch.load(os.path.join(CFG['save_path'], 'best_model.pth'))
# model = BaseModel()
model.load_state_dict(checkpoint)
model.eval()

preds = predict(model, test_loader, device)

test_df['label'] = preds
test_df.to_csv(f'./submit/{CFG["submit_file"]}', index=False)








