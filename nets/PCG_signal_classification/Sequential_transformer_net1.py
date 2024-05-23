import torch
import torch.nn as nn
import math
import torch.nn.functional as F


"""
    Graph Transformer
    
"""
from layers.mlp_readout_layer import MLPReadout

class PositionalEncoding(nn.Module):
    """
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """

    def __init__(self, d_model, vocab_size=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(vocab_size, d_model)
        position = torch.arange(0, vocab_size, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:,: x.size(1), :]
        return self.dropout(x)

class TransformerNet(nn.Module):

    def __init__(self, net_params):
        super().__init__()

        d_model = net_params['in_dim'] # node_dim (feat is an integer)
        hidden_dim = net_params['hidden_dim']

        n_classes = net_params['n_classes']
        num_heads = net_params['n_heads']

        dropout = net_params['dropout']
        n_layers = net_params['L']

        self.n_classes = n_classes
        self.device = net_params['device']


        token_size = net_params['sigLen']
        # d_model-> in_dim;
        self.pos_encoder = PositionalEncoding(
            d_model=d_model,
            dropout=dropout,
            vocab_size=token_size,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
        )
        # self.classifier = nn.Linear(d_model, n_classes)
        self.MLP_layer = MLPReadout(d_model, n_classes)
        self.d_model = d_model


    def forward(self, h):

        # output
        h = self.pos_encoder(h)
        h = self.transformer_encoder(h)

        h = self.MLP_layer(h)
        # h_out = torch.sigmoid(h)
        return h
    
    def loss(self, pred, label):
        label = label.reshape(-1)
        pred = pred.view(pred.shape[0]*pred.shape[1], -1)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, label)

        return loss

    # def loss(self, pred, label):
    #
    #     # calculating label weights for weighted loss computation
    #     label = label.view(-1, label.shape[2]).squeeze(dim=1)
    #     pred = pred.view(-1, pred.shape[2])
    #     V = label.size(0)
    #     # label_count = torch.bincount(label.to(torch.float))
    #     label_count = torch.bincount(label.to(torch.int32))
    #     label_count = label_count[label_count.nonzero()].squeeze()
    #     cluster_sizes = torch.zeros(self.n_classes).long().to(self.device)
    #     cluster_sizes[torch.unique(label)] = label_count
    #     weight = (V - cluster_sizes).float() / V
    #     weight *= (cluster_sizes>0).float()
    #
    #     # weighted cross-entropy for unbalanced classes
    #     criterion = nn.CrossEntropyLoss(weight=weight)
    #     loss = criterion(pred, label)
    #
    #     return loss