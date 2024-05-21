import math
import time
import scipy.io
import torch
import numpy as np
class load_splitDataSet(torch.utils.data.Dataset):

    def __init__(self,
                 data,
                 split, sigLength = 2000, fet_dim =132):

        #  save('mat2torch.mat', 'fsstTrain', 'trainLabels', 'fsstVal', 'valLabels', 'fsstTest', 'testLabels')
        self.L, self.W = sigLength, fet_dim
        if split == 'train':
            self.data = self.preparedata(data['fsstTrain'])
            self.label = self.preparelabel(data['trainLabels'])
            # self.data = self.preparedata(data['fsstVal'])
            # self.label = self.preparelabel(data['valLabels'])

        elif  split == 'val':
            self.data = self.preparedata(data['fsstVal'])
            self.label = self.preparelabel(data['valLabels'])
        elif split == 'test':
            self.data = self.preparedata(data['fsstTest'])
            self.label = self.preparelabel(data['testLabels'])

        self.n_samples = int(len(self.data)/self.L)
        # self.n_samples = len(self.data)


    def preparedata(self, data):
        N = len(data)
        t = np.reshape(data, newshape=(N))

        ten_data =torch.from_numpy(t[0])
        ten_data = torch.transpose(ten_data, 0, 1).float()
        for id in range(1, N):
            z = torch.from_numpy(t[id])
            # ten_data=torch.concat((ten_data, torch.transpose(z, 0, 1).float()), dim=0)
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

        # sigs = sigs.view(1000, 64, 3)
        # sigs = sigs.permute(2, 0, 1)
        return sigs, labs
        # sigs = self.data[idx]
        # labs = self.label[idx]
        # # labels = labs.view(-1, labs.shape[1]).squeeze(dim=1)
        # return sigs, labs




class pcgDataset(torch.utils.data.Dataset):

    def __init__(self, name, sigLength = 2000, fet_dim =132):
        """
            Loading SBM datasets
        """
        start = time.time()
        print("[I] Loading dataset %s..." % (name))
        self.name = name
        # data_dir = 'data/signal/'
        data_dir = 'D://workspace//ECG//fPCG//data2//'
        self.L = sigLength
        data = scipy.io.loadmat((data_dir + name+'.mat'))
        # data = obj['fsstTrain', 'trainLabels', 'fsstVal', 'valLabels', 'fsstTest', 'testLabels']
        self.PCG = {}

        for split in [ 'train','val', 'test']: #
            self.PCG[split] = load_splitDataSet(data, split, sigLength, fet_dim)

    # form a mini batch from a given list of samples = [(graph, label) pairs]
    def collate(self, samples):
        # The input samples is a list of pairs (graph, label).
        signals, labels = map(list, zip(*samples))

        # sig_list, label_list = [], []
        # lengths = len(labels)
        # for id in range(0, lengths):
        #     # st = self.L * id
        #     # et = self.L * (id+1)
        #     one_cut = signals[id]
        #     sig_list.append(one_cut)
        #     one_cut = labels[id]
        #     label_list.append(one_cut)
        #
        # batch_sig = torch.as_tensor(signals, dtype=torch.float32)
        # batch_label = torch.as_tensor(labels, dtype=torch.long)
        batch_sig = torch.stack(signals)
        batch_label = torch.stack(labels)
        return batch_sig, batch_label
        # return signals, labels
    
    




