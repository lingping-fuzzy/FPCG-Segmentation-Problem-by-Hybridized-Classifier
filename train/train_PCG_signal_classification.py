"""
    Utility functions for training one epoch 
    and evaluating one epoch
"""
import torch
import torch.nn as nn

from train.metrics import accuracy_sigs as accuracy

def train_epoch(model, optimizer, device, data_loader, epoch):

    model.train()
    epoch_loss = 0
    epoch_train_acc = 0

    for iter, (batch_sigs, batch_labels) in enumerate(data_loader):
        batch_sigs = batch_sigs.to(device)

        batch_labels = batch_labels.to(device)
        optimizer.zero_grad()

        batch_scores = model.forward(batch_sigs)

        loss = model.loss(batch_scores, batch_labels)
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5) # new added
        optimizer.step()
        epoch_loss += loss.detach().item()
        num_size = batch_labels.shape[1]*batch_labels.shape[0]
        epoch_train_acc += accuracy(batch_scores, batch_labels)/num_size
    epoch_loss /= (iter + 1)
    epoch_train_acc /= (iter + 1)
    
    return epoch_loss, epoch_train_acc, optimizer


def evaluate_network(model, device, data_loader, epoch):
    
    model.eval()
    epoch_test_loss = 0
    epoch_test_acc = 0

    with torch.no_grad():
        for iter, (batch_sigs, batch_labels) in enumerate(data_loader):
            batch_sigs = batch_sigs.to(device)
            batch_labels = batch_labels.to(device)

            batch_scores = model.forward(batch_sigs)
            loss = model.loss(batch_scores, batch_labels) 
            epoch_test_loss += loss.detach().item()

            num_size = batch_labels.shape[1] * batch_labels.shape[0]
            epoch_test_acc += accuracy(batch_scores, batch_labels)/num_size
        epoch_test_loss /= (iter + 1)
        epoch_test_acc /= (iter + 1)
        
    return epoch_test_loss, epoch_test_acc


