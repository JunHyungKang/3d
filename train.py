import os
from tqdm.auto import tqdm
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from data import CustomDataset
from val import validation


def train(model, optimizer, train_loader, val_loader, scheduler, device, writer, args):
    save_path = f'./weights/{args.submit_csv.split(".")[0]}'
    model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    best_score = 0
    for epoch in range(1, args.num_epoch + 1):
        model.train()
        train_loss = []

        for data, label in tqdm(iter(train_loader)):
            data, label = data.float().to(device), label.long().to(device)
            optimizer.zero_grad()

            output = model(data)
            loss = criterion(output, label)

            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

        if scheduler is not None:
            scheduler.step()

        val_loss, val_acc = validation(model, criterion, val_loader, device)
        print(f'Epoch : [{epoch}] Train Loss : [{np.mean(train_loss)}] Val Loss : [{val_loss}] Val ACC : [{val_acc}]')
        writer.add_scalar('Param/lr', optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar('Loss/train', np.mean(train_loss), epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('val_acc', val_acc, epoch)

        if best_score < val_acc:
            best_score = val_acc
            os.makedirs(save_path, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(save_path, 'best_model.pth'))
