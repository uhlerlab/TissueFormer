import torch
from torch_geometric.data import Data, Dataset

class CustomDataset(Dataset):
    def __init__(self, datasets):
        """
        Initializes the dataset with a list of graphs. Each graph is a dictionary containing:
        - 'x': node features (Tensor of shape [num_nodes, num_features])
        - 'edge_index': edge list (Tensor of shape [2, num_edges])
        - 'y': node labels (Tensor of shape [num_nodes])
        """
        self.datasets = datasets
        super(CustomDataset, self).__init__()

    def len(self):
        return len(self.datasets)

    def get(self, idx):
        """
        Get a single graph based on the index.
        Returns a PyTorch Geometric Data object.
        """
        dataset = self.datasets[idx]

        # Create a PyTorch Geometric Data object
        data = Data(
            x=dataset['x'],  # Node features
            y=dataset['y']  # Node labels
        )
        if 'x2' in dataset.keys():
            data.x2 = dataset['x2']
        if 'gene_idx' in dataset.keys():
            data.gene_idx = dataset['gene_idx']
        else:
            data.gene_idx = None
        if 'hvg_gene_rank' in dataset.keys():
            data.hvg_gene_rank = dataset['hvg_gene_rank']
        else:
            data.hvg_gene_rank = torch.zeros(data.y.shape[1], dtype=torch.long)
        if 'gene_eval_mask' in dataset.keys():
            data.gene_eval_mask = dataset['gene_eval_mask']
            assert data.hvg_gene_rank.shape[0] == data.gene_eval_mask.sum()
        if 'microphage_mask' in dataset.keys():
            data.cell_mask = dataset['microphage_mask']
        if 'cell_mask' in dataset.keys():
            data.cell_mask = dataset['cell_mask']

        if 'edge_index' in dataset.keys():
            data.edge_index = dataset['edge_index']
        else:
            data.edge_index = None
        if 'edge_weight' in dataset.keys():
            data.edge_weight = dataset['edge_weight']
        else:
            data.edge_weight = None
        if 'split_idx' in dataset.keys():
            data.split_idx = dataset['split_idx']
        else:
            data.split_idx = None
        return data

    def merge(self, new_data):
        self.datasets += new_data