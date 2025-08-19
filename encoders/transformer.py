import math,os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import degree

def full_attention_conv(qs, ks, vs, output_attn=False):
    '''
    qs: query tensor [N, H, M]
    ks: key tensor [L, H, M]
    vs: value tensor [L, H, D]

    return output [N, H, D]
    '''
    # normalize input
    qs = qs / torch.norm(qs, p=2, dim=-1, keepdim=True) # [N, H, M]
    ks = ks / torch.norm(ks, p=2, dim=-1, keepdim=True) # [L, H, M]
    N = qs.shape[0]

    # numerator
    kvs = torch.einsum("lhm,lhd->hmd", ks, vs)
    attention_num = torch.einsum("nhm,hmd->nhd", qs, kvs) # [N, H, D]
    all_ones = torch.ones([vs.shape[0]]).to(vs.device)
    vs_sum = torch.einsum("l,lhd->hd", all_ones, vs) # [H, D]
    attention_num += vs_sum.unsqueeze(0).repeat(vs.shape[0], 1, 1) # [N, H, D]

    # denominator
    all_ones = torch.ones([ks.shape[0]]).to(ks.device)
    ks_sum = torch.einsum("lhm,l->hm", ks, all_ones)
    attention_normalizer = torch.einsum("nhm,hm->nh", qs, ks_sum)  # [N, H]

    # attentive aggregated results
    attention_normalizer = torch.unsqueeze(attention_normalizer, len(attention_normalizer.shape))  # [N, H, 1]
    attention_normalizer += torch.ones_like(attention_normalizer) * N
    attn_output = attention_num / attention_normalizer # [N, H, D]

    # compute attention for visualization if needed
    if output_attn:
        # attention = (torch.einsum("nhm,lhm->nlh", qs, ks) + 1) / attention_normalizer.transpose(1, 2) # [N, L, H]
        attention = torch.einsum("nhm,lhm->nlh", qs, ks) # [N, L, H]

    if output_attn:
        return attn_output, attention
    else:
        return attn_output

def graph_convolution_conv(x, edge_index, edge_weight):
    N = x.shape[0]
    row, col = edge_index
    d = degree(col, N).float()
    d_norm_in = (1. / d[col]).sqrt()
    d_norm_out = (1. / d[row]).sqrt()
    if edge_weight is None:
        value = torch.ones_like(row) * d_norm_in * d_norm_out
    else:
        value = edge_weight * d_norm_in * d_norm_out
    value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
    adj = torch.sparse_coo_tensor(torch.stack([col, row], dim=0), value, [N, N])
    gcn_conv_output = torch.sparse.mm(adj, x) # [N, D]
    return gcn_conv_output

class TransConv(nn.Module):
    '''
    one Transformer layer
    '''
    def __init__(self, in_channels, out_channels, num_heads, use_graph=True, use_residual=True):
        super(TransConv, self).__init__()
        self.Wk = nn.Linear(in_channels, out_channels * num_heads)
        self.Wq = nn.Linear(in_channels, out_channels * num_heads)
        self.Wv = nn.Linear(in_channels, out_channels * num_heads)
        self.Wg = nn.Linear(in_channels, out_channels)
        self.Wr = nn.Linear(in_channels, out_channels)

        self.out_channels = out_channels
        self.num_heads = num_heads
        self.use_graph = use_graph
        self.use_residual = use_residual

    def reset_parameters(self):
        self.Wk.reset_parameters()
        self.Wq.reset_parameters()
        self.Wv.reset_parameters()
        self.Wg.reset_parameters()
        self.Wr.reset_parameters()

    def forward(self, query_input, source_input, edge_index=None, edge_weight=None, output_attn=False):
        # feature transformation
        query = self.Wq(query_input).reshape(-1, self.num_heads, self.out_channels)
        key = self.Wk(source_input).reshape(-1, self.num_heads, self.out_channels)
        value = self.Wv(source_input).reshape(-1, self.num_heads, self.out_channels)

        # compute full attentive aggregation
        if output_attn:
            attention_output, attn = full_attention_conv(query, key, value, output_attn)  # [N, H, D]
        else:
            attention_output = full_attention_conv(query, key, value) # [N, H, D]

        final_output = attention_output.mean(dim=1) # [N, D]

        # use input graph for gcn conv
        if self.use_graph:
            gnn_input = self.Wg(source_input)
            gnn_output = graph_convolution_conv(gnn_input, edge_index, edge_weight)
            final_output += gnn_output
        if self.use_residual:
            final_output += self.Wr(source_input)

        if output_attn:
            return final_output, attn.mean(dim=-1)
        else:
            return final_output

class Transformer(nn.Module):
    '''
    Transformer model class
    x: input node features [N, D]
    edge_index: 2-dim indices of edges [2, E]
    return y_hat predicted logits [N, C]
    '''
    def __init__(self, in_channels, hidden_channels, num_layers_prop=3, num_layers_mlp=3, num_attn_heads=1,
                 dropout=0., use_bn=True, use_graph=True, use_residual=True):
        super(Transformer, self).__init__()

        layers_in = [nn.Linear(in_channels, hidden_channels), nn.BatchNorm1d(hidden_channels), nn.ELU(), nn.Dropout(dropout)]
        for _ in range(num_layers_mlp - 1):
            layers_in.append(nn.Linear(hidden_channels, hidden_channels))
            layers_in.append(nn.BatchNorm1d(hidden_channels))
            layers_in.append(nn.ELU())
            layers_in.append(nn.Dropout(dropout))

        self.mlp_in = nn.Sequential(*layers_in)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(num_layers_prop):
            self.convs.append(TransConv(hidden_channels, hidden_channels, num_heads=num_attn_heads, use_graph=use_graph, use_residual=use_residual))
            # self.bns.append(nn.BatchNorm1d(hidden_channels))
            self.bns.append(nn.LayerNorm(hidden_channels))

        self.dropout = dropout
        self.activation = F.elu
        self.use_bn = use_bn

    def reset_parameters(self):
        for layer in self.mlp_in.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index=None, edge_weight=None):
        layer_ = []

        # input MLP layer
        x = self.mlp_in(x)

        # store as residual link
        layer_.append(x)

        for i, conv in enumerate(self.convs):
            # graph convolution with DIFFormer layer
            x = conv(x, x, edge_index, edge_weight)
            x = x + layer_[-1]
            if self.use_bn:
                x = self.bns[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            layer_.append(x)

        return x

    def get_embeddings(self, x, edge_index=None, edge_weight=None):
        layer_ = []
        x = self.mlp_in(x)
        layer_.append(x)
        for i, conv in enumerate(self.convs):
            x = conv(x, x, edge_index, edge_weight)
            x = x + layer_[-1]
            if self.use_bn:
                x = self.bns[i](x)
            x = self.activation(x)
            layer_.append(x)
        return torch.stack(layer_, dim=0) # [layer num, N, D]

    def get_attentions(self, x, edge_index=None, edge_weight=None):
        layer_, attentions = [], []
        x = self.mlp_in(x)
        layer_.append(x)
        for i, conv in enumerate(self.convs):
            x, attn = conv(x, x, edge_index, edge_weight, output_attn=True)
            x = x + layer_[-1]
            attentions.append(attn)
            if self.use_bn:
                x = self.bns[i](x)
            x = self.activation(x)
            layer_.append(x)
        return torch.stack(attentions, dim=0) # [layer num, N, N]