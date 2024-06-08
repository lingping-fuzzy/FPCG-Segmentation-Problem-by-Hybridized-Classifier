"""
    IMPORTING LIBS
"""

import argparse, json
import pickle


"""
    IMPORTING CUSTOM MODULES/METHODS
"""

from data.data import LoadData 

def main(task = None):
    """
        USER CONTROLS
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help="Please give a config.json file with training/model/data/param details")
    parser.add_argument('--dataset', help="Please give a value for dataset name")

    parser.add_argument('--sigLen', help="Please give a value for gpu id")
    parser.add_argument('--fet_dim', help="Please give a value for model name")
    parser.add_argument('--n_classes', help="Please give a value for dataset name")


    args = parser.parse_args()
    args.config = "conf/PCG_LapPE_Classification.json"

    with open(args.config) as f:
        config = json.load(f)
        
    DATASET_NAME = 'mat2torch1'
    DATASET_NAME1 = 'mat2torch2'

    # network parameters
    net_params = config['net_params']

    if args.sigLen is not None:
        net_params['sigLen'] = int(args.sigLen)
    if args.fet_dim is not None:
        net_params['fet_dim'] = int(args.fet_dim)
    if args.n_classes is not None:
        net_params['n_classes'] = int(args.n_classes)


    # dataset = LoadData(DATASET_NAME, net_params, DATASET_NAME1)
    # print('finish loading' )
    # file = open('signal_data.pkl', 'wb')
    # import pickle
    # pickle.dump(dataset, file)
    # file.close()
    

    from data.data import LoadMixDataXBG
    dataset = LoadMixDataXBG(DATASET_NAME, net_params, DATASET_NAME1)
    print('finish loading' )
    file = open('xgbsignal_data.pkl', 'wb')
    import pickle
    pickle.dump(dataset, file)
    file.close()


if __name__ == "__main__":
    # this is for training a model
    task = 'training'
    # task = 'working'

    main(task)


























