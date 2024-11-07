import torch

def run_pretrain(model, dataloader, optimizer, device):
    model.train()
    running_loss = 0.

    for dataset in dataloader:
        data = dataset[0]
        x1, x2 = data.x.to(device), data.y.to(device)
        edge_index = data.edge_index.to(device)
        gene_idx = data.gene_idx.to(device)
        optimizer.zero_grad()
        loss = model.loss(x1, x2, gene_idx, edge_index)
        loss.backward()
        running_loss += loss.item()
        optimizer.step()

    return running_loss / len(dataloader)