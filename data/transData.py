import math
import time
import scipy.io
import torch
import numpy as np
class loadnn_splitDataSet(torch.utils.data.Dataset):

    def __init__(self,
                 data,
                 split, sigLength = 1000, fet_dim =192):

        #  save('mat2torch.mat', 'fsstTrain', 'trainLabels', 'fsstVal', 'valLabels', 'fsstTest', 'testLabels')
        self.L, self.W = sigLength, fet_dim
        if split == 'train':
            self.data = self.preparedata(data['fsstTrain'])
            self.label = self.preparelabel(data['trainLabels'])
        self.W = self.data.shape[1]
        self.n_samples = int(len(self.data)/self.L)


    def preparedata(self, data):
        N = len(data)
        t = np.reshape(data, newshape=(N))

        ten_data =torch.from_numpy(t[0])
        ten_data = torch.transpose(ten_data, 0, 1).float()
        for id in range(1, N):
            z = torch.from_numpy(t[id])
            ten_data = torch.concat((ten_data, torch.transpose(z, 0, 1).float()), dim=0)
        return ten_data

    def preparelabel(self, data):
        N = len(data)
        t = np.reshape(data, newshape=(N))
        ten_data =torch.from_numpy(t[0] -1)
        ten_data = torch.transpose(ten_data, 0, 1)
        for id in range(1, N):
            z = torch.from_numpy(t[id] -1)
            ten_data = torch.concat((ten_data, torch.transpose(z, 0, 1).long()), dim=0)
        ten_data = ten_data.squeeze(-1)
        return ten_data


    def __len__(self):
        """Return the number of graphs in the dataset."""
        return self.n_samples

    def __getitem__(self, idx):
        """
            Get the idx^th sample.
            Parameters
            ---------
            idx : int
                The sample index.
            Returns
            -------
        """
        # idx = int(math.floor(idx/2000))
        st = self.L * idx
        et = self.L *(idx+1)
        sigs = self.data[st:et]
        labs = self.label[st:et]
        return sigs, labs

class TransDataNN(torch.utils.data.Dataset):

    def __init__(self, name, sigLength = 1000, fet_dim =192):
        """
            Loading SBM datasets
        """
        start = time.time()
        print("[I] Loading dataset %s..." % (name))
        self.name = name
        data_dir = 'D:\\workspace\\ECG\\fPCG\\data2\\' #'data/mat/'
        self.L = sigLength
        data = scipy.io.loadmat((data_dir + name+'.mat'))
        # data = obj['fsstTrain', 'trainLabels', 'fsstVal', 'valLabels', 'fsstTest', 'testLabels']
        self.PCG = {}
        for split in [ 'train']: # for testing data, we only use 'train'
            self.PCG[split] = loadnn_splitDataSet(data, split, sigLength, fet_dim)
            print('finish ', split)

    # form a mini batch from a given list of samples = [(graph, label) pairs]
    def collate(self, samples):
        # The input samples is a list of pairs (graph, label).
        signals, labels = map(list, zip(*samples))

        batch_sig = torch.stack(signals)
        batch_label = torch.stack(labels)
        return batch_sig, batch_label
    
    
class loadxgb_splitDataSet(torch.utils.data.Dataset):

    def __init__(self,
                 data,
                 split, sigLength=2000, fet_dim=132):

        self.L, self.W = sigLength, fet_dim
        if split == 'train':
            self.data = self.preparedata(data['fsstTrain'])
            self.label = self.preparelabel(data['trainLabels'])

        self.n_samples = len(self.data)

    def preparedata(self, data):
        N = len(data)
        t = np.reshape(data, newshape=(N))

        ten_data =t[0]
        for id in range(1, N):
            ten_data = np.concatenate((ten_data, t[id]), axis=1)
        return np.transpose(ten_data)

    def preparelabel(self, data):
        N = len(data)
        t = np.reshape(data, newshape=(N))

        ten_data =np.squeeze(t[0] - 1)
        for id in range(1, N):
            ten_data = np.concatenate((ten_data, np.squeeze(t[id]-1)), axis=0)

        return np.int64(ten_data)

    def __len__(self):
        """Return the number of graphs in the dataset."""
        return self.n_samples

    def __getitem__(self, idx):
        """
            Get the idx^th sample.
            Parameters
            ---------
            idx : int
                The sample index.
            Returns
            -------
        """
        sigs = self.data[idx]
        labs = self.label[idx]
        return sigs, labs


class TransDataXGB(torch.utils.data.Dataset):

    def __init__(self, name, sigLength=2000, fet_dim=132):
        """
            Loading SBM datasets
        """
        start = time.time()
        print("[I] Loading dataset %s..." % (name))
        self.name = name
        data_dir = 'D:\\workspace\\ECG\\fPCG\\data2\\' #'data/mat/'
        self.L = sigLength
        data = scipy.io.loadmat((data_dir + name + '.mat'))
        self.PCG = {}
        for split in ['train']:
            self.PCG[split] = loadxgb_splitDataSet(data, split, sigLength, fet_dim)



