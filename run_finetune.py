import torch
from metrics import calculate_metrics, calculate_score

def run_update(model, dataloader, device, use_gene_idx=True):
    model.eval()
    for dataset in dataloader:
        data = dataset[0]
        inputs, labels = data.x.to(device), data.y.to(device)
        if labels.is_sparse:
            labels = labels.to_dense()
        edge_index = data.edge_index.to(device)
        if use_gene_idx:
            gene_idx = data.gene_idx.to(device)
        else:
            gene_idx = None
        train_idx = data.split_idx['train_idx'].to(device)

        model.update(inputs, labels, gene_idx, train_idx, edge_index)

def run_train(model, dataloader, criterion, optimizer, device, args, use_gene_idx=True):
    model.train()
    train_loss = 0.

    for i, dataset in enumerate(dataloader):
        data = dataset[0]
        inputs, labels = data.x.to(device), data.y.to(device)
        if labels.is_sparse:
            labels = labels.to_dense()
        edge_index = data.edge_index.to(device)
        if use_gene_idx:
            gene_idx = data.gene_idx.to(device)
        else:
            gene_idx = None
        train_idx = data.split_idx['train_idx'].to(device)

        if args.evaluate_task in ['niche_classification', 'cell_type_classification'] and args.method == 'ours-MLP':
            inputs2 = data.x2.to(device)
            outputs = model(inputs, inputs2, gene_idx, edge_index)
        elif args.evaluate_task == 'region_time_prediction':
            inputs2 = data.x2.to(device)
            group_idx = data.group_idx.to(device)
            outputs = model(inputs, inputs2, gene_idx, edge_index, group_idx=group_idx)
        elif args.evaluate_task == 'gene_regression':
            outputs = model(inputs, gene_idx, edge_index)
        else: # classification
            outputs = model(inputs, edge_index)

        if hasattr(data, 'gene_eval_mask'):
            gene_eval_mask = data.gene_eval_mask.to(device)
            labels, outputs = labels[:, gene_eval_mask], outputs[:, gene_eval_mask]

        if isinstance(criterion, torch.nn.NLLLoss):
            outputs = torch.nn.functional.log_softmax(outputs, dim=1)
        if isinstance(criterion, torch.nn.BCEWithLogitsLoss):
            if len(labels.shape) <= 1:
                labels = labels.reshape(-1, 1)
            labels = labels.float()

        if args.evaluate_task in ['he_annotation_classification', 'niche_classification', 'cell_type_classification']:
            cell_mask = data.cell_mask.to(device)
            if cell_mask.sum() == 0:
                continue
            loss = criterion(outputs[cell_mask], labels[cell_mask])
        elif args.evaluate_task == 'region_time_prediction':
            outputs, labels = outputs[1:], labels[1:]
            if labels.shape[0] == 0:
                continue
            loss = criterion(outputs, labels)
        else:
            loss = criterion(outputs[train_idx], labels[train_idx])

        loss.backward()
        train_loss += loss.item()

        optimizer.step()
        optimizer.zero_grad()

    return train_loss / len(dataloader)

@torch.no_grad()
def run_test(model, dataloader, device, args, use_gene_idx=True):
    test_metric = []
    n = len(dataloader)
    model.eval()
    for dataset in dataloader:
        data = dataset[0]
        inputs, labels = data.x.to(device), data.y.to(device)
        if labels.is_sparse:
            labels = labels.to_dense()
        edge_index = data.edge_index.to(device)
        if use_gene_idx:
            gene_idx = data.gene_idx.to(device)
        else:
            gene_idx = None
        test_idx = data.split_idx['test_idx'].to(device)

        if args.evaluate_task in ['niche_classification', 'cell_type_classification'] and args.method == 'ours-MLP':
            inputs2 = data.x2.to(device)
            outputs = model(inputs, inputs2, gene_idx, edge_index)
        elif args.evaluate_task == 'region_time_prediction':
            inputs2 = data.x2.to(device)
            group_idx = data.group_idx.to(device)
            outputs = model(inputs, inputs2, gene_idx, edge_index, group_idx=group_idx)
        elif args.evaluate_task == 'gene_regression':
            outputs = model(inputs, gene_idx, edge_index)
        else:  # classification
            outputs = model(inputs, edge_index)

        if hasattr(data, 'gene_eval_mask'):
            gene_eval_mask = data.gene_eval_mask.to(device)
            labels_eval, outputs_eval = labels[:, gene_eval_mask], outputs[:, gene_eval_mask]
        else:
            labels_eval, outputs_eval = labels, outputs

        if args.evaluate_task == 'gene_regression':
            hvg_gene_rank = data.hvg_gene_rank.to(device)
            hvg_gene_mask = (hvg_gene_rank < args.hvg_gene_top)
            metric = calculate_score(labels_eval, outputs_eval, 'RMSE', test_idx, hvg_gene_mask)
        elif args.evaluate_task in ['he_annotation_classification', 'niche_classification', 'cell_type_classification']:
            cell_mask = data.cell_mask.to(device)
            if cell_mask.sum() == 0:
                continue
            outputs_pos_eval = outputs_eval[:, 1]
            metric = calculate_score(labels_eval, outputs_pos_eval, 'AUC', cell_mask)
        elif args.evaluate_task == 'region_time_prediction':
            labels_eval, outputs_eval = labels_eval[1:], outputs_eval[1:]
            if labels_eval.shape[0] == 0:
                continue
            test_idx = torch.ones(labels_eval.shape[0], dtype=bool)
            metric = calculate_score(labels_eval, outputs_eval, 'RMSE', test_idx)
        elif args.evaluate_task == 'macrophage_identification':
            metric = calculate_score(labels_eval, outputs_eval, 'AUC', test_idx)
        elif args.evaluate_task == 'macrophage_classification':
            cell_mask = data.cell_mask.to(device)
            metric = calculate_score(labels_eval, outputs_eval, 'Accuracy_TopK', cell_mask, class_num=args.cell_type_num)

        test_metric += [metric]

    return sum(test_metric) / len(test_metric)

@torch.no_grad()
def evaluate(model, dataloader, device, args, use_gene_idx=True, output_result=False):
    if args.evaluate_task == 'gene_regression':
        running_metrics = [0.] * len(args.metrics) * len(args.hvg_gene_tops)
    else:
        running_metrics = [0.] * len(args.metrics)
    model.eval()
    n = 0
    num_metrics = len(args.metrics)

    y_pred, y_true = [], []
    for dataset in dataloader:
        data = dataset[0]
        inputs, labels = data.x.to(device), data.y.to(device)
        if labels.is_sparse:
            labels = labels.to_dense()
        edge_index = data.edge_index.to(device)
        if use_gene_idx:
            gene_idx = data.gene_idx.to(device)
        else:
            gene_idx = None
        test_idx = data.split_idx['test_idx'].to(device)
        hvg_gene_rank = data.hvg_gene_rank.to(device)

        if args.evaluate_task in ['niche_classification', 'cell_type_classification'] and args.method == 'ours-MLP':
            inputs2 = data.x2.to(device)
            outputs = model(inputs, inputs2, gene_idx, edge_index)
        elif args.evaluate_task == 'region_time_prediction':
            inputs2 = data.x2.to(device)
            group_idx = data.group_idx.to(device)
            outputs = model(inputs, inputs2, gene_idx, edge_index, group_idx=group_idx)
        elif args.evaluate_task == 'gene_regression':
            outputs = model(inputs, gene_idx, edge_index)
        else:  # classification
            outputs = model(inputs, edge_index)

        if hasattr(data, 'gene_eval_mask'):
            gene_eval_mask = data.gene_eval_mask.to(device)
            labels_eval, outputs_eval = labels[:, gene_eval_mask], outputs[:, gene_eval_mask]
        else:
            labels_eval, outputs_eval = labels, outputs

        if args.evaluate_task == 'gene_regression':
            for t in range(len(args.hvg_gene_tops)):
                hvg_gene_mask = (hvg_gene_rank<args.hvg_gene_tops[t])
                metrics = calculate_metrics(labels_eval, outputs_eval, args.metrics, test_idx, hvg_gene_mask)
                for m in range(num_metrics):
                    running_metrics[num_metrics*t+m] = running_metrics[num_metrics*t+m] + metrics[m]
        elif args.evaluate_task in ['he_annotation_classification', 'niche_classification', 'cell_type_classification']:
            test_idx = data.cell_mask.to(device)
            if test_idx.sum() == 0:
                continue
            outputs_pos_eval = outputs_eval[:, 1]
            metrics = calculate_metrics(labels_eval, outputs_pos_eval, args.metrics, test_idx, class_num=args.cell_type_num)
            for m in range(num_metrics):
                running_metrics[m] = running_metrics[m] + metrics[m]
        elif args.evaluate_task == 'region_time_prediction':
            labels_eval, outputs_eval = labels_eval[1:], outputs_eval[1:]
            if labels_eval.shape[0] == 0:
                continue
            test_idx = torch.ones(labels_eval.shape[0], dtype=bool)
            metrics = calculate_metrics(labels_eval, outputs_eval, args.metrics, test_idx, class_num=args.cell_type_num)
            for m in range(num_metrics):
                running_metrics[m] = running_metrics[m] + metrics[m]

        n += 1
        if output_result:
            y_pred += [outputs_eval]
            y_true += [labels_eval]

    if output_result:
        y_pred = torch.cat(y_pred, dim=0)
        y_true = torch.cat(y_true, dim=0)
        return y_pred, y_true
    else:
        running_metrics = [r / n for r in running_metrics]
        return running_metrics