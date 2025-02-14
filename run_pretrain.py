import torch

def run_pretrain(model, dataloader, optimizer, device, accumulate_steps):
    model.train()
    running_loss = 0.

    for i, dataset in enumerate(dataloader):
        data = dataset[0]
        x1, x2 = data.x.to(device), data.y.to(device)
        edge_index = data.edge_index.to(device)
        gene_idx = data.gene_idx.to(device)
        train_idx = data.split_idx['train_idx'].to(device)

        loss = model.loss(x1, x2, gene_idx, train_idx, edge_index)
        loss.backward()
        running_loss += loss.item()

        if (i + 1) % accumulate_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

    return running_loss / len(dataloader)