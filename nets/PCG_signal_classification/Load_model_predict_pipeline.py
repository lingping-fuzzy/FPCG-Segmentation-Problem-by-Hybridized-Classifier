import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import confusion_matrix
import math
import torch.nn.functional as F

"""
    Graph Transformer

"""

from nets.PCG_signal_classification.load_net import pcg_model
from torch.utils.data import DataLoader


class loodNet(nn.Module):

    def __init__(self, MODEL_NAME, dataset, params, net_params, dir):
        super().__init__()

        self.model = pcg_model(MODEL_NAME, net_params)
        self.dataset = dataset
        self.CM = np.zeros([4,4])
        self.params = params
        self.weight = None
        self.dir = dir


    def prepare(self):
        # output
        # parms = torch.load(self.dir, map_location=torch.device('cpu'))
        self.model.load_state_dict(torch.load(self.dir))
        self.model.eval()
        self.pre_predict()

    def pre_predict(self):
        testset = self.dataset.PCG['val']  # it should be 'val'
        test_loader = DataLoader(testset, batch_size=self.params['batch_size'], shuffle=False, collate_fn=self.dataset.collate)
        self.model.eval()
        for iter, (batch_sigs, batch_labels) in enumerate(test_loader):
            batch_scores = self.model.forward(batch_sigs)
            self.confusion_sigs(batch_scores, batch_labels)

        self.weight = self.calculate_weight_trans()

    def confusion_sigs(self, scores, targets):
        targets = targets.reshape(-1)
        scores = scores.view(scores.shape[0] * scores.shape[1], -1)
        S = targets.cpu().numpy()
        C = np.argmax(torch.nn.Softmax(dim=1)(scores).cpu().detach().numpy(), axis=1)
        self.CM = self.CM + confusion_matrix(S, C).astype(np.float32)

    def calculate_weight(self):
        total = np.sum(self.CM, axis=0)
        dia = self.CM.diagonal()
        weights = dia/total
        return weights

    def calculate_weight_trans(self):
        temp = torch.ones([4, 4]) * -1
        temp.fill_diagonal_(1)
        total = np.sum(self.CM, axis=0)
        test1 = self.CM / total
        test2 = torch.mul(temp, torch.from_numpy(test1))
        weights = torch.sum(test2, dim=0)
        print('This is the accuracy of DNN')
        print(self.CM)
        return weights


    def predict_prob(self, dataset):
        testset = self.dataset.PCG['test']  # it should be 'test'
        test_loader = DataLoader(testset, batch_size=self.params['batch_size'], shuffle=False, collate_fn=self.dataset.collate)
        self.model.eval()
        preds = []
        for iter, (batch_sigs, batch_labels) in enumerate(test_loader):
            batch_scores = self.model.forward(batch_sigs)
            batch_scores = batch_scores.view(batch_scores.shape[0] * batch_scores.shape[1], -1)
            preds.append(batch_scores)
        return preds