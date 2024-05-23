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

        # dmodel = net_params['in_dim'] # node_dim (feat is an integer)
        hidden_dim = net_params['hidden_dim']

        n_classes = net_params['n_classes']
        num_heads = net_params['n_heads']

        dropout = net_params['dropout']
        n_layers = net_params['L']

        self.n_classes = n_classes
        self.device = net_params['device']
        self.batch_size = net_params['batch_size']
        self.in_dim = net_params['sigLen']
        self.HWorder = net_params['HWorder']
        self.kerSize = net_params['kernel_size']
        self.convlayer1 = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=8, kernel_size=self.kerSize, padding=1),
            nn.BatchNorm1d(8),
            # nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.convlayer2 = nn.Sequential(
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=1, padding=1),
            nn.BatchNorm1d(16),
            # nn.ReLU()
            nn.MaxPool1d(2)
        )
        # if self.kerSize == 3:
        #     d_model = 4 * 17  # node_dim (feat is an integer)
        # else:
        #     d_model = 4 * 17*2  # node_dim (feat is an integer)
        if self.kerSize == 3:
            d_model = 16 * 12  # node_dim (feat is an integer)
        else:
            d_model = 16 * 12  # node_dim (feat is an integer)
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
        L, W, Z = h.shape
        if self.HWorder == True:
            # h = h.reshape([L*W, 1, 64, 3]) # for singlen=192
            h = h.reshape([L * W, 2, 44])
        else:
            # h = h.reshape([L * W, 1, 3, 64])# for singlen=192
            h = h.reshape([L * W, 2, 44])

        h = self.convlayer1(h)

        h = self.convlayer2(h)
        if self.kerSize == 3:
            # h = h.squeeze(dim=3)
            # h = h.view(L, W, 4 * 17) # for singlen=192
            h = h.view(L, W, 16 * 12)
        else:
            # h = h.view(L, W, 4 * 17*2) # for singlen=192
            h = h.view(L, W, 16 * 12)

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