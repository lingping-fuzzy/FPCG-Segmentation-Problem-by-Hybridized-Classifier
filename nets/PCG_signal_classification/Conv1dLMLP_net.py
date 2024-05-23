import torch
import torch.nn as nn
import math
import torch.nn.functional as F

"""
    Graph Transformer

"""
from layers.mlp_readout_layer import MLPReadout


class ConvL1dMLPNet(nn.Module):

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
            self.act = nn.Softmax(dim=2)
        elif net_params['act_fun'] == 'leakyRelu':
            self.act = nn.LeakyReLU()

        self.kerSize = net_params['kernel_size']
##https://pythonguides.com/pytorch-conv1d/

        if layer_norm == False and batch_norm == False:
            self.convlayer1 = nn.Sequential(
                nn.Conv1d(in_channels=64, out_channels=32, kernel_size=self.kerSize, padding=1),
                nn.Conv1d(in_channels=32, out_channels=16, kernel_size=1, padding=1),
                self.act,
                nn.MaxPool2d(kernel_size=2, stride=2)
            )

        elif layer_norm == True and batch_norm == False:
            self.convlayer1 = nn.Sequential(
                nn.Conv1d(in_channels=64, out_channels=32, kernel_size=self.kerSize, padding=1),
                nn.LayerNorm([32, 5]),
                nn.Conv1d(in_channels=32, out_channels=16, kernel_size=1, padding=1),
                self.act,
                nn.MaxPool2d(kernel_size=2, stride=2)
            )

        elif layer_norm == False and batch_norm == True:
            self.convlayer1 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=self.kerSize, padding=1),
            nn.BatchNorm1d(32),
            nn.Conv1d(in_channels=32, out_channels=16, kernel_size=1, padding=1),
            self.act,
            nn.MaxPool2d(kernel_size=2, stride=2)
            )
            # self.l1 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=self.kerSize, padding=1),
            # self.l2 = nn.BatchNorm1d(32),
            # self.l3 = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=1, padding=1),
            # self.l4 = self.act,
            # self.l5 = nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            self.convlayer1 = nn.Sequential(
                nn.Conv1d(in_channels=64, out_channels=32, kernel_size=self.kerSize, padding=1),
                nn.BatchNorm1d(32),
                self.act,
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            self.convlayer2 = nn.Sequential(
                nn.Conv1d(in_channels=32, out_channels=16, kernel_size=1, padding=1),
                nn.BatchNorm2d(16),
                self.act,
                nn.MaxPool2d(2)
            )

        self.dropout = nn.Dropout(p=dropout)
        if self.kerSize == 3:
            d_model = 16  # node_dim (feat is an integer)
        elif layer_norm == True and batch_norm == True:
            d_model = 32
        # elif layer_norm == False and batch_norm == True:
        #     d_model = 64
        else:
            d_model = 24  # node_dim (feat is an integer)

        self.resual_lin = nn.Linear(feature_size, d_model)
        self.MLP_layer = MLPReadout(d_model, n_classes)
        self.d_model = d_model

    def forward(self, h):
        L, W, Z = h.shape  # batch_size; 1000-signal-length;  Z-feature size
        residual = h
        if self.HWorder == True:
            h = h.reshape([L * W, 64, 3])
        else:
            h = h.reshape([L * W, 3, 64])

        h = self.convlayer1(h)
        # h = self.l1(h)
        # h = self.l2(h)
        # h = self.l3(h)
        # h = self.l4(h)
        # h = self.l5(h)

        h = h.view(L, W, h.shape[1] * h.shape[2])

        residual = self.resual_lin(residual)
        h = self.dropout(h)
        if self.residual == True:
            h = residual + h
        h = self.MLP_layer(h)

        return h

    def loss(self, pred, label):
        label = label.reshape(-1)
        pred = pred.view(pred.shape[0] * pred.shape[1], -1)
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