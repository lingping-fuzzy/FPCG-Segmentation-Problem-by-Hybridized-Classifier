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

class ConvMLPNet(nn.Module):

    def __init__(self, net_params):
        super().__init__()

        # dmodel = net_params['in_dim'] # node_dim (feat is an integer)
        self.residual = net_params['residual']

        n_classes = net_params['n_classes']
        batch_norm = net_params['batch_norm']

        layer_norm = net_params['layer_norm']
        dropout = net_params['dropout']
        self.HWorder = net_params['HWorder']
        self.n_classes = n_classes
        self.device = net_params['device']
        self.batch_size = net_params['batch_size']
        self.in_dim = net_params['sigLen']
        feature_size = net_params['fet_dim']
        if net_params['act_fun'] == 'relu':
            self.act = nn.ReLU()
        elif net_params['act_fun'] == 'softmax':
            self.act = nn.Softmax(dim=3)
        elif net_params['act_fun'] == 'leakyRelu':
            self.act = nn.LeakyReLU()
        self.kerSize = net_params['kernel_size']
        if self.kerSize  == 3:
            layer_norm1 = nn.LayerNorm([3, 64, 3])
            layer_norm2 = nn.LayerNorm([4, 34, 3])
        elif self.kerSize  == 1:
            layer_norm1 = nn.LayerNorm([3, 66, 5])
            layer_norm2 = nn.LayerNorm([4, 35, 4])

        if layer_norm == True:
            self.convlayer1 = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=3, kernel_size=self.kerSize , padding=1),
                layer_norm1,
                self.act,
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            self.convlayer2 = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=4, kernel_size=1, padding=1),
                layer_norm2,
                self.act,
                nn.MaxPool2d(2)
            )
        if batch_norm == True:
            self.convlayer1 = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=3, kernel_size=self.kerSize , padding=1),
                nn.BatchNorm2d(3),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            self.convlayer2 = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=4, kernel_size=1, padding=1),
                nn.BatchNorm2d(4),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )

        self.dropout = nn.Dropout(p= dropout)
        if self.kerSize == 3:
            d_model = 4 * 17  # node_dim (feat is an integer)
        else:
            d_model = 4 * 17*2  # node_dim (feat is an integer)
        self.linear_res = nn.Linear(d_model, feature_size)
        self.MLP_layer = MLPReadout(feature_size, n_classes)
        self.d_model = d_model


    def forward(self, h):
        L, W, Z = h.shape  #batch_size; 1000-signal-length;  Z-feature size
        residual = h
        if self.HWorder == True:
            h = h.reshape([L*W, 1, 64, 3])
        else:
            h = h.reshape([L * W, 1, 3, 64])

        h = self.convlayer1(h)
        h = self.convlayer2(h)

        if self.kerSize == 3:
            h = h.squeeze(dim=3)
            h = h.view(L, W, 4 * 17)
        else:
            h = h.view(L, W, 4 * 17*2)

        h = self.linear_res(h)
        h = self.dropout(h)
        if self.residual == True:
            h = residual + h
        h = self.MLP_layer(h)

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