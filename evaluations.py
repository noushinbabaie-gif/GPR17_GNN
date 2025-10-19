import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import r2_score, mean_absolute_error
from scipy import stats


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr, data.batch, data.u)
        loss = F.mse_loss(out, data.y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)

def evaluate(model, loader, device):
    model.eval()
    total_loss = 0
    preds = []
    trues = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.edge_attr, data.batch, data.u)
            loss = F.mse_loss(out, data.y.view(-1))
            total_loss += loss.item() * data.num_graphs
            preds.append(out.cpu().numpy())
            trues.append(data.y.view(-1).cpu().numpy())
    avg_loss = total_loss / len(loader.dataset)
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)

    # Compute R², slope, intercept
    r2 = r2_score(trues, preds)
    slope, intercept, r_value, p_value, std_err = stats.linregress(trues, preds)
    return avg_loss, r2, slope, intercept, preds, trues

def evaluate_statistics_for_datasets(model, train_loader, val_loader, test_loader, device):
    """
    Computes typical regression statistics (MSE, R², slope, intercept) 
    for three datasets: train, val, test.
    Returns a dictionary of results.
    """
    results = {}
    for name, loader in zip(['Train', 'Val', 'Test'], [train_loader, val_loader, test_loader]):
        avg_loss, r2, slope, intercept, preds, trues = evaluate(model, loader, device)
        mae = mean_absolute_error(trues, preds)
        results[name] = {
            'MSE': avg_loss,
            'MAE': mae,
            'R2': r2,
            'Slope': slope,
            'Intercept': intercept
        }
    return results
