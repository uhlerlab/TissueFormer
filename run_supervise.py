import torch
from evaluation import calculate_metrics, calculate_mse

def run_train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.

    for dataset in dataloader:
        data = dataset[0]
        inputs, labels = data.x.to(device), data.y.to(device)
        if labels.is_sparse:
            labels = labels.to_dense()
        edge_index = data.edge_index.to(device)
        train_idx = data.split_idx['train_idx'].to(device)
        optimizer.zero_grad()
        outputs = model(inputs, edge_index)
        loss = criterion(outputs[train_idx], labels[train_idx])
        loss.backward()
        running_loss += loss.item()
        optimizer.step()

    return running_loss / len(dataloader)

@torch.no_grad()
def run_valid(model, dataloader, criterion, device):
    running_loss, running_mse = 0., 0.
    model.eval()
    for dataset in dataloader:
        data = dataset[0]
        inputs, labels = data.x.to(device), data.y.to(device)
        if labels.is_sparse:
            labels = labels.to_dense()
        edge_index = data.edge_index.to(device)
        valid_idx = data.split_idx['valid_idx'].to(device)
        hvg_gene_mask = data.hvg_gene_mask.to(device)
        outputs = model(inputs, edge_index)
        loss = criterion(outputs[valid_idx], labels[valid_idx])
        running_mse += calculate_mse(labels, outputs, valid_idx, hvg_gene_mask)
        running_loss += loss.item()
    return running_loss / len(dataloader), running_mse / len(dataloader)

@torch.no_grad()
def run_test(model, dataloader, criterion, device):
    running_loss, running_mse = 0., 0.
    model.eval()
    for dataset in dataloader:
        data = dataset[0]
        inputs, labels = data.x.to(device), data.y.to(device)
        if labels.is_sparse:
            labels = labels.to_dense()
        edge_index = data.edge_index.to(device)
        test_idx = data.split_idx['test_idx'].to(device)
        hvg_gene_mask = data.hvg_gene_mask.to(device)
        outputs = model(inputs, edge_index)
        loss = criterion(outputs[test_idx], labels[test_idx])
        running_mse += calculate_mse(labels, outputs, test_idx, hvg_gene_mask)
        running_loss += loss.item()
    return running_loss / len(dataloader), running_mse / len(dataloader)

@torch.no_grad()
def evaluate(model, dataloader, device):
    running_metrics = [0., 0., 0., 0.]
    model.eval()
    n = len(dataloader)
    for dataset in dataloader:
        data = dataset[0]
        inputs, labels = data.x.to(device), data.y.to(device)
        if labels.is_sparse:
            labels = labels.to_dense()
        edge_index = data.edge_index.to(device)
        test_idx = data.split_idx['test_idx'].to(device)
        hvg_gene_mask = data.hvg_gene_mask.to(device)
        outputs = model(inputs, edge_index)
        metrics = calculate_metrics(labels, outputs, test_idx, hvg_gene_mask)
        running_metrics = [running_metrics[i] + metrics[i] / n for i in range(len(running_metrics))]
    return running_metrics