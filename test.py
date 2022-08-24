from tqdm.auto import tqdm

import torch

def predict(model, test_loader, device):
    model.to(device)
    model.eval()
    model_preds = []
    with torch.no_grad():
        for data in tqdm(iter(test_loader)):
            data = data.float().to(device)

            batch_pred = model(data)

            model_preds += batch_pred.argmax(1).detach().cpu().numpy().tolist()

    return model_preds

