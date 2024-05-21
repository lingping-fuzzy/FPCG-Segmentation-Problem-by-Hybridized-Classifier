"""
    File to load dataset based on user control from main file
"""
from data.signal import pcgDataset
from data.xgbsignal import XGBpcgDataset
from data.mixSignal import DatasetCom
from data.mixxgbsignal import XGBmixpcgDataset
from data.transData import TransDataNN, TransDataXGB

def LoadData(DATASET_NAME, net_params =None, extraName=None):
    """
        This function is called in the main_xx.py file 
        returns:
        ; dataset object
    """    
    # handling for fPCG signal dataset
    sig_DATASETS = 'PCG_signal'
    if extraName is not None:
        return DatasetCom(DATASET_NAME, extraName, net_params['sigLen'], net_params['fet_dim'])

    if DATASET_NAME == 'mat2torch':
        return pcgDataset(DATASET_NAME, net_params['sigLen'], net_params['fet_dim'])


def LoadDataXBG(DATASET_NAME, net_params =None):
    return XGBpcgDataset(DATASET_NAME, net_params['sigLen'], net_params['fet_dim'])

def LoadMixDataXBG(DATASET_NAME, net_params =None, extraName=None):
    return XGBmixpcgDataset(DATASET_NAME, extraName, net_params['sigLen'], net_params['fet_dim'])

def transData(DATASET_NAME, net_params =None, source = None):
    if source == 'nn':
        return TransDataNN(DATASET_NAME, net_params['sigLen'], net_params['fet_dim'])
    elif source == 'xgb':
        return TransDataXGB(DATASET_NAME, net_params['sigLen'], net_params['fet_dim'])