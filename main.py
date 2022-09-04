import os
import shutil
import h5py # .h5 파일을 읽기 위한 패키지
import random
import pandas as pd
import numpy as np
import argparse

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models.basemodel import BaseModel
import models.c3d
import models.shufflenetv2
import models.squeezenet
import models.resnext
import models.resnet
import models.mobilenet
import models.mobilenetv2
import models.shufflenet
from train import train
from data import CustomDataset
from test import predict

import warnings
warnings.filterwarnings(action='ignore')
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f'{device} will work for this task')


def parse_args():
    parser = argparse.ArgumentParser(description='3d classification')
    parser.add_argument('--num_epoch', type=int, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, help='Number of batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate')
    parser.add_argument('--input_size', type=int, help='3d tensor size')
    parser.add_argument('--seed', type=int, help='seed for reproducing')
    parser.add_argument('--train_csv', type=str, help='path for train csv file')
    parser.add_argument('--valid_csv', type=str, help='path for valid csv file')
    parser.add_argument('--test_csv', type=str, help='path for test csv file')
    parser.add_argument('--trainval_data', type=str, help='path for trainval array')
    parser.add_argument('--test_data', type=str, help='path for test array')
    parser.add_argument('--submit_sample', type=str, help='path for submit_sample')
    parser.add_argument('--submit_csv', type=str, help='name for submit file')
    parser.add_argument('--model', type=str, help='basemodel, c3d')
    parser.add_argument('--scheduler', type=str, help='Exponential, Cyclic')
    args = parser.parse_args()
    return args


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


if __name__ == '__main__':
    args = parse_args()
    seed_everything(args.seed)

    if os.path.isdir(f'./runs/{args.submit_csv.split(".")[0]}'):
        shutil.rmtree(f'./runs/{args.submit_csv.split(".")[0]}')
    writer = SummaryWriter(f'./runs/{args.submit_csv.split(".")[0]}')

    if args.model == 'basemodel':
        model = BaseModel()
    elif args.model == 'c3d':
        from models.c3d import get_fine_tuning_parameters
        model = models.c3d.get_model(
            num_classes=10,
            sample_size=args.input_size,
            sample_duration=args.input_size)
    elif args.model == 'mobilenet':
        model = models.mobilenet.get_model(
            num_classes=10,
            sample_size=args.input_size,
            width_mult=args.input_size)
    elif args.model == 'mobilenetv2':
        model = models.mobilenetv2.get_model(
            num_classes=10,
            sample_size=args.input_size,
            width_mult=args.input_size)
    elif args.model == 'resnet':
        model = models.resnet.resnet152(
            num_classes=10,
            sample_size=args.input_size,
            sample_duration=args.input_size)
    elif args.model == 'resnext':
        model = models.resnext.resnext152(
            num_classes=10,
            sample_size=args.input_size,
            sample_duration=args.input_size)
    # elif args.model == 'shufflenet':
    #     model = models.shufflenet.get_model(
    #         num_classes=10,
    #         sample_size=args.input_size,
    #         width_mult=args.input_size)
    elif args.model == 'shufflenetv2':
        model = models.shufflenetv2.get_model(
            num_classes=10,
            sample_size=args.input_size,
            width_mult=args.input_size)
    elif args.model == 'squeezenet':
        model = models.squeezenet.get_model(
            num_classes=10,
            sample_size=args.input_size,
            sample_duration=args.input_size)


    model.eval()

    if args.scheduler == 'Exponential':
        optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96, last_epoch=-1)
    elif args.scheduler == 'Cyclic':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer=optimizer, base_lr=args.lr/100,
                                                      max_lr=args.lr, step_size_up=10,
                                                      step_size_down=None, mode='exp_range',
                                                      gamma=0.995, scale_fn=None, scale_mode='cycle',
                                                      cycle_momentum=True, base_momentum=0.8, max_momentum=0.9,
                                                      last_epoch=- 1, verbose=False)

    # TODO: 5 split + 앙상블 코드 추가
    train_df = pd.read_csv(args.train_csv)
    train_dataset = CustomDataset(train_df['ID'].values, train_df['label'].values, augment=True, task='trainval',
                                  args=args)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    val_df = pd.read_csv(args.valid_csv)
    val_dataset = CustomDataset(val_df['ID'].values, val_df['label'].values, augment=True, task='trainval', args=args)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # TODO: tensorboard를 wandb로 변경
    train(model, optimizer, train_loader, val_loader, scheduler, device, writer, args)

    test_df = pd.read_csv(args.submit_sample)
    test_dataset = CustomDataset(test_df['ID'].values, None, augment=False, task='test', args=args)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    checkpoint = torch.load(os.path.join(f'./weights/{args.submit_csv.split(".")[0]}/best_model.pth'))
    model.load_state_dict(checkpoint)
    model.eval()

    preds = predict(model, test_loader, device)

    test_df['label'] = preds
    test_df.to_csv(f'./submit/{args.submit_csv}', index=False)








