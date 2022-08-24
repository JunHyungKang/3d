from tqdm.auto import tqdm
import numpy as np

import torch
from sklearn.metrics import accuracy_score


def validation(model, criterion, val_loader, device):
    model.eval()
    true_labels = []
    model_preds = []
    val_loss = []
    with torch.no_grad():
        for data, label in tqdm(iter(val_loader)):
            data, label = data.float().to(device), label.long().to(device)

            model_pred = model(data)
            loss = criterion(model_pred, label)

            val_loss.append(loss.item())

            # TODO: 실제 output 확인 (1epoch accuracy가 너무 높고, test 성능과 차이가 많이 남)
            model_preds += model_pred.argmax(1).detach().cpu().numpy().tolist()
            true_labels += label.detach().cpu().numpy().tolist()

    return np.mean(val_loss), accuracy_score(true_labels, model_preds)