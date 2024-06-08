import torch
import numpy as np
from main_pcg_signal_classification import main as DNN_main
from main_pcg_signal_classification import nnModel_predict
from main_pcg_Tree_classification import main as tree_main
from main_pcg_Tree_classification import xgbModel_predict
from sklearn.metrics import confusion_matrix
import pickle
import pandas as pd
from pretty_confusion_matrix import pp_matrix
import matplotlib.pyplot as plt


# name = ['mASD1_data', 'AEN_data', 'AWGN_data', 'AMFN_data',
#         'AEN_AMFN_data', 'AEN_AWGN_data', 'AWGN_AMFN_data', 'mASD1_AEN_data', 'mASD1_AMFN_data', 'mASD1_AWGN_data',
#         'AEN_AWGN_AMFN_data', 'mASD1_AEN_AMFN_data', 'mASD1_AEN_AWGN_data', 'mASD1_AWGN_AMFN_data']

# name = ['AEN_test_data', 'AMFN_test_data', 'AWGN_test_data', 'mASD1_test_data']
def confusion_res(scores, targets):
    targets = targets.reshape(-1)
    S = targets
    C = np.argmax(torch.nn.Softmax(dim=1)(scores).cpu().detach().numpy(), axis=1)
    CM = confusion_matrix(S, C).astype(np.float32)
    total = np.sum(CM, axis=0)
    prob = CM / total
    return CM, prob

def create_torchData():
    file = 'torchsignal.pkl'
    dataset = None
    with open(file, 'wb') as handle:
        pickle.dump(dataset, handle)

def getXGBoostPred():
    task = 'working'
    datatrain_name='xgb-300'
    tree_preds, tree_weight, testlabel = tree_main(task)
    CM, prob = confusion_res(torch.from_numpy(tree_preds), testlabel.label)
    cmap = 'PuRd'
    df_cm = pd.DataFrame(CM, index=range(1, 5), columns=range(1, 5))
    figname = datatrain_name+'-tree.png'
    pp_matrix(df_cm, cmap=cmap, savetitle=figname)
    print(CM)
    print(prob)
    print('finish loading')
    file = open(('data/'+datatrain_name+'-preds.pkl'), 'wb')
    pickle.dump(tree_preds, file)
    file.close()
    file = open(('data/'+datatrain_name+'-weights.pkl'), 'wb')
    pickle.dump(tree_weight, file)
    file.close()

def getNeuralNNPred():
    task = 'working'
    datatrain_name='nn-5layer'
    #Transformer_PCG_signal1_GPU1_08h18m46s_on_Jan_15_2023  #909  --4layer
    #Transformer_PCG_signal1_GPU1_13h52m58s_on_Jan_16_2023 #673  --3layer
    # Transformer_PCG_signal1_GPU0_13h53m52s_on_Jan_16_2023  # 773  --5layer
    conffile = 'GPU0_13h53m52s_on_Jan_16_2023'  # 773
    DNN_preds, DNN_weight, testlabel = DNN_main(task, conffile)
    z = torch.stack(DNN_preds[:-1])
    z1 = z.view(z.shape[0] * z.shape[1], -1)
    z2 = torch.stack(DNN_preds[-1:])
    z2 = z2.view(z2.shape[0] * z2.shape[1], -1)
    z3 = torch.cat((z1, z2), dim=0)
    DNN_ = torch.sigmoid(z3)
    CM, prob = confusion_res(DNN_, testlabel)
    cmap = 'PuRd'
    df_cm = pd.DataFrame(CM, index=range(1, 5), columns=range(1, 5))
    figname = datatrain_name+'-nn.png'
    pp_matrix(df_cm, cmap=cmap, savetitle=figname)
    print(CM)
    print(prob)
    print('finish loading')
    file = open(('data/'+datatrain_name+'-preds.pkl'), 'wb')
    pickle.dump(DNN_preds, file)
    file.close()
    file = open(('data/'+datatrain_name+'-weights.pkl'), 'wb')
    pickle.dump(DNN_preds, file)
    file.close()

def instanceCombine(conffile):
    task = 'working'
    tree_preds, tree_weight, testlabel = tree_main(task)

    CM, prob = confusion_res(torch.from_numpy(tree_preds), testlabel.label)
    cmap = 'PuRd'
    df_cm = pd.DataFrame(CM, index=range(1, 5), columns=range(1, 5))
    figname = 'xgb-400-tree.png'
    pp_matrix(df_cm, cmap=cmap, savetitle=figname)
    print(CM)
    print(prob)
    #option use one--use the model generate the results
    # DNN_preds, DNN_weight, testlabel = DNN_main(task, conffile)
    # # DNN_weight = tensor([0.9142, 0.8366, 0.8832, 0.8105], dtype=torch.float64)
    # print('This is the accuracy of combination model')

    # # option use two--use the saved data to obtain the results
    # file = open("data//mat//tree-preds-realdata-test.pkl", 'rb')
    # import pickle
    # tree_preds = pickle.load(file)
    # file.close()
    # tree_weight=[0.99336619, 0.98462538, 0.99334199, 0.84896879]
    # tree_weight = np.array(tree_weight)
    #
    # file = open("data//mat//nn-RD-preds.pkl", 'rb')
    # DNN_preds = pickle.load(file)
    # file.close()
    # # DNN_weight=[0.9142, 0.8366, 0.8832, 0.8105]
    # # DNN_weight = np.array(DNN_weight)
    # DNN_weight = torch.tensor([0.9142, 0.8366, 0.8832, 0.8105], dtype=torch.float64)
    #
    # file = open("data//mat//RD-label.pkl", 'rb')
    # testlabel = pickle.load(file)
    # file.close()
    #
    #
    # z = torch.stack(DNN_preds[:-1])
    # z1 = z.view(z.shape[0] * z.shape[1], -1)
    # z2 = torch.stack(DNN_preds[-1:])
    # z2 = z2.view(z2.shape[0] * z2.shape[1], -1)
    # z3 = torch.cat((z1, z2), dim=0)
    # DNN_ = torch.sigmoid(z3)
    #
    # total_weight = DNN_weight + tree_weight
    # DNN_weight = DNN_weight/total_weight
    # tree_weight = tree_weight/total_weight
    #
    # DNN_p = torch.mul(DNN_, DNN_weight)
    # tree_p = torch.from_numpy(tree_preds)*tree_weight
    # final_p = DNN_p+tree_p
    #
    # names = ['tree', 'NN', 'comb']
    # for method in names:
    #     if method == 'tree':
    #         input = tree_p
    #     elif method == 'NN':
    #         input = DNN_p
    #     elif method == 'comb':
    #         input = final_p
    #     CM, prob = confusion_res(input, testlabel)
    #     print('this is for ', method)
    #     print(torch.from_numpy(CM))
    #     print(torch.from_numpy(prob))
    #
    #     cmap = 'PuRd'
    #     df_cm = pd.DataFrame(CM, index=range(1, 5), columns=range(1, 5))
    #     figname = 'RD-test'+'_'+method+'.png'
    #     pp_matrix(df_cm, cmap=cmap, savetitle=figname)


if __name__ == '__main__':
    task = 'modelAndData'
    # conffile = 'GPU0_13h00m37s_on_Nov_27_2022'
    # conffile = 'GPU2_11h32m27s_on_Dec_07_2022'
    # conffile = 'GPU2_15h24m29s_on_Nov_27_2022'
    # conffile = 'GPU1_17h16m23s_on_Dec_01_2022'
    # conffile = 'GPU2_22h40m53s_on_Dec_07_2022'

    # getXGBoostPred()
    getNeuralNNPred()
    ## real data
    # conffile = 'GPU2_08h16m35s_on_Jan_02_2023'
    # instanceCombine(conffile)





    # DNN_model, DNN_weight, _ = DNN_main(task, conffile)
    # tree_model, tree_weight, _ = tree_main(task)
    #
    # namesALL = ['one', 'two', 'three', 'four']#
    # types = ['mASD1', 'AEN', 'AMFN', 'AWGN']
    #
    # DNN_weight[DNN_weight < 0] = 0
    # tree_weight[tree_weight< 0] = 0
    # total_weight = DNN_weight + tree_weight
    # DNN_weight = DNN_weight / total_weight
    # tree_weight = tree_weight / total_weight
    #
    # all_cm = []
    # all_prob = []
    # for type in types:
    #     for idname in namesALL:
    #         file = open(('data/mat/' + 'nn' + type + '-' + idname + '.pkl'), 'rb')
    #         nn_dataset = pickle.load(file)
    #         file.close()
    #
    #         file = open(('data/mat/'+'xgb'+type+'-'+idname+'.pkl'), 'rb')
    #         tree_dataset = pickle.load(file)
    #         file.close()
    #
    #         tree_preds = xgbModel_predict(tree_model, tree_dataset)
    #         DNN_preds = nnModel_predict(DNN_model, nn_dataset)
    #         print('This is the accuracy of combination model')
    #         z = torch.stack(DNN_preds[:-1])
    #         z1 = z.view(z.shape[0] * z.shape[1], -1)
    #         z2 = torch.stack(DNN_preds[-1:])
    #         z2 = z2.view(z2.shape[0] * z2.shape[1], -1)
    #         z3 = torch.cat((z1, z2), dim=0)
    #         DNN_ = torch.sigmoid(z3)
    #         # DNN_preds = torch.stack(DNN_preds)
    #         # DNN_preds = torch.sigmoid(DNN_preds)
    #
    #         # DNN_ = DNN_preds.view(DNN_preds.shape[0] * DNN_preds.shape[1], -1)
    #         DNN_p = torch.mul(DNN_, DNN_weight)
    #         tree_p = torch.from_numpy(tree_preds) * tree_weight
    #         final_p = DNN_p + tree_p
    #         pall_cm=[]
    #         pall_prob=[]
    #         names = ['tree', 'NN', 'comb']
    #         for method in names:
    #             if method == 'tree':
    #                 input = tree_p
    #             elif method == 'NN':
    #                 input = DNN_p
    #             elif method == 'comb':
    #                 input = final_p
    #             CM, prob = confusion_res(input, tree_dataset.PCG['train'].label)
    #             # print('this is for ', method)
    #             # cmap = 'PuRd'
    #             # df_cm = pd.DataFrame(CM, index=range(1, 5), columns=range(1, 5))
    #             # figname = 'GPU1Dec1DEec9'+type+'-'+idname+'_'+method+'.png'
    #             # pp_matrix(df_cm, cmap=cmap, savetitle=figname)
    #             pall_cm.append(CM)
    #             pall_prob.append(prob)
    #         all_cm.append(np.concatenate((pall_cm[0], pall_cm[1], pall_cm[2]), axis=1))
    #         all_prob.append(np.concatenate((pall_prob[0], pall_prob[1], pall_prob[2]), axis=1))
    #
    # test = np.stack(all_cm)
    # test1 = np.reshape(test, (test.shape[0] * test.shape[1], test.shape[2]))
    # np.savetxt('CMvalues.csv', test1, delimiter=",")
    # print(np.stack(all_cm))
    #
    # test = np.stack(all_prob)
    # test1 = np.reshape(test, (test.shape[0] * test.shape[1], test.shape[2]))
    # np.savetxt('Probvalue.csv', test1, delimiter=",")


