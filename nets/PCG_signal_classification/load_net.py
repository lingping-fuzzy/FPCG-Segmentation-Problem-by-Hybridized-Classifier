"""
    Utility file to select GraphNN model as
    selected by the user
"""

from nets.PCG_signal_classification.Sequential_transformer_net import TransformerNet
from nets.PCG_signal_classification.ConvMLP_net import ConvMLPNet
from nets.PCG_signal_classification.Conv1dMLP_net import Conv1DMLPNet
from nets.PCG_signal_classification.Conv1dLMLP_net import ConvL1dMLPNet


def Transformers(net_params):
    return TransformerNet(net_params)

def ConvMLP(net_params):
    return ConvMLPNet(net_params)

def conv1dlMLP(net_params):
    return ConvL1dMLPNet(net_params)

def conv1dMLP(net_params):
    return Conv1DMLPNet(net_params)

def pcg_model(MODEL_NAME, net_params):
    models = {
        'Transformer': Transformers,
        'convMLP': ConvMLP,
        'conv1dMLP':conv1dMLP,
        'Lconv1dMLP': conv1dlMLP,
    }
        
    return models[MODEL_NAME](net_params)