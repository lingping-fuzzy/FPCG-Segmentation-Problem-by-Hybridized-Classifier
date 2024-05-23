"""
    Utility functions for training one epoch 
    and evaluating one epoch
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import confusion_matrix
import sys
# %logstart -o "test.log"
import time
logtime = open(("runlogtime.txt"), "a")


# clear explain what is xgboost and xgbClassifier
# https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/
# reference from: https://mljar.com/blog/xgboost-save-load-python/
# https://xgboost.readthedocs.io/en/stable/python/python_api.html?highlight=cv#xgboost.cv
# n_estimators (int) – Number of gradient boosted trees. Equivalent to number of boosting rounds.#
# but this n_estimators should be in xgb.cv and as a parameter, not in params, though params also includes a 'num_boost_round'

class train_process():
    def __init__(self, data_loader, net_params):
        self.data_loader = data_loader
        self.data_dmatrix = None
        self.res = None
        self.net_params = net_params
# use 'multi:softprob' gives probability and use 'multi:softmax' gives the class label
    def fit(self, x):
        params = {'objective': 'multi:softprob',
                  'eval_metric': 'mlogloss',
                  'num_class': 4,
                  'eta': x[0],
                  'subsample': x[1],
                  'max_depth': int(x[2])}
        xgb_cv = xgb.cv(dtrain=self.data_dmatrix, params=params,
                        nfold=4,  seed=self.net_params['seed'],
                        num_boost_round=self.net_params['n_estimators'])
        print(xgb_cv[-1:].values[0])
        return xgb_cv[-1:].values[0]

    def train_epoch(self):
        X_train, y_train = self.data_loader.data, self.data_loader.label
        # eta [default=0.3, alias: learning_rate]
        self.data_dmatrix = xgb.DMatrix(data=X_train, label=y_train)
        grid = pd.DataFrame({'eta': [0.01, 0.05, 0.1] * 3,
                             'subsample': np.repeat([0.3, 0.5, 0.8], 3),
                             "max_depth": np.repeat(np.int_([4]), 9)})

        grid[['train-loss-mean', 'train-loss-std',
              'test-loss-mean', 'test-loss-std']] = grid.apply(self.fit, axis=1, result_type='expand')

        self.res = grid.to_string()

class LoadModel():
    def __init__(self, dir, data_loader):
        self.xgb_model = xgb.Booster()
        self.CM = np.zeros([4,4])
        self.load_model(dir=dir)
        self.weights_trans( data_loader)
        self.weight = self.calculate_weight_trans()

    def load_model(self, dir):
        self.xgb_model.load_model((dir + "//model_sklearn.json"))

    def weights_trans(self, data_loader):

        test = xgb.DMatrix(data_loader.data, np.int64(data_loader.label))
        if self.xgb_model.best_ntree_limit is None:
            preds = self.xgb_model.predict(test)
        else:
            preds = self.xgb_model.predict(test)

        yhat_labels = np.argmax(preds, axis=1)
        self.CM = confusion_matrix(data_loader.label, yhat_labels)
        print('This is the accuracy of tree')
        print(self.CM)

    def predict_prob(self, data_loader):

        test = xgb.DMatrix(data_loader.data, np.int64(data_loader.label))
        preds = self.xgb_model.predict(test)
        # preds = self.xgb_model.predict(test, ntree_limit=self.xgb_model.best_ntree_limit)

        return preds

    def calculate_weight(self):
        total = np.sum(self.CM, axis=0)
        dia = self.CM.diagonal()
        weights = dia/total
        return weights
    def calculate_weight_trans(self):
        temp = np.ones((4, 4))*-1
        np.fill_diagonal(temp, 1)
        total = np.sum(self.CM, axis=0)
        propt = self.CM / total
        nptest2 = np.multiply(temp, propt)
        weights = np.sum(nptest2, axis=0)

        return weights



import logging
logger = logging.getLogger(__name__)

class XGBLogging(xgb.callback.TrainingCallback):
    """log train logs to file"""

    def __init__(self, epoch_log_interval=100):
        self.epoch_log_interval = epoch_log_interval
        self.start_time = time.time()

    def after_iteration(self, model, epoch, evals_log):
        if epoch % self.epoch_log_interval == 0:
            for data, metric in evals_log.items():
                metrics = list(metric.keys())
                metrics_str = ""
                for m_key in metrics:
                    metrics_str = metrics_str + f"{m_key}: {metric[m_key][-1]}"
                logger.info(f"Epoch: {epoch}, {data}: {metrics_str}")
                costtime = time.time()-self.start_time
                #logtime.write("Time record {0}: {1}\n".format(str(epoch), str(costtime)))
                print('here1')
                with open(logtime, 'a') as f:
                    f.write("Time record {0}: {1}\n".format(str(epoch), str(costtime)))
                print('here2')

        # False to indicate training should not stop.
        return False


class SetModel():
    def __init__(self, net_params, dir=None):
        self.eta = net_params['eta']
        self.subsample = net_params['subsample']
        self.max_depth = net_params['max_depth']
        self.xgb_model = None
        self.net_params = net_params
        if dir != None:
            sys.stdout = open((dir+'\\loss.log'), 'a')

    def fit_trans(self, train_loader, val_loader):
        X_train, y_label = train_loader.data, train_loader.label
        X_test, y_test = val_loader.data, val_loader.label

        train = xgb.DMatrix(data=X_train, label=np.int64(y_label))
        val = xgb.DMatrix(data=X_test, label=np.int64(y_test))
        # https://mljar.com/blog/xgboost-save-load-python/
        params = {'objective': 'multi:softprob',
                  'eta': self.eta,
                  'subsample': self.subsample,
                  'max_depth': self.max_depth,
                  'num_class': 4,
                  'eval_metric': 'mlogloss'}

        self.xgb_model = xgb.train(params, train, evals=[(train, "train"), (val, "validation")],
                                   num_boost_round=self.net_params['n_estimators'], early_stopping_rounds=10, callbacks=[XGBLogging(epoch_log_interval=1)])
        #logtime.close()

    def predict_trans(self, data_loader):

        test = xgb.DMatrix(data_loader.data, np.int64(data_loader.label))
        preds = self.xgb_model.predict(test, ntree_limit =self.xgb_model.best_ntree_limit)
        yhat_labels = np.argmax(preds, axis=1)
        from sklearn.metrics import confusion_matrix
        print(confusion_matrix(data_loader.label, yhat_labels))
        return preds, yhat_labels

    def predict_trans1(self, data_loader):
        X_, y_ = data_loader.data, data_loader.label
        preds = self.xgb_model.predict(X_, ntree_limit=self.xgb_model.best_ntree_limit)
        return preds

    def save_model(self, dir):
        self.xgb_model.save_model((dir + "//model_sklearn.json"))

    def load_model(self, dir):
        model = xgb.Booster()
        model.load_model((dir + "//model_sklearn.json"))
        return model


class oldSetModel():
    def __init__(self, net_params):
        self.eta = net_params['eta']
        self.subsample = net_params['subsample']
        self.max_depth = net_params['max_depth']
        self.xgb_model = None
        self.net_params = net_params

    def fit_trans1(self, train_loader, val_loader):
        X_train, y_train = train_loader.data, train_loader.label
        X_test, y_test = val_loader.data, val_loader.label
        from sklearn.preprocessing import OneHotEncoder
        labels = OneHotEncoder(sparse=False).fit_transform(y_train.reshape(-1, 1))
        # train = xgb.DMatrix(data=train_loader.data, label=train_loader.label)
        # val = xgb.DMatrix(data=val_loader.data, label=val_loader.label)
        # https://mljar.com/blog/xgboost-save-load-python/
        params = {'objective': 'binary:logistic',
                  'eval_metric': 'map',
                  'eta': self.eta,
                  'subsample': self.subsample,
                  'max_depth': self.max_depth,}
        xgb_model = xgb.XGBClassifier(objective='reg:squaredlogerror',
                                           eval_metric='map',
                                           eta=self.eta,
                                           subsample=self.subsample,
                                           max_depth=self.max_depth, n_estimators=self.net_params['n_estimators'],
                                           seed=self.net_params['seed'])
        # self.xgb_model = xgb.XGBClassifier(params, train, evals=[(train, "train"), (val, "validation")],
        #                            num_boost_round=self.net_params['n_estimators'],
        #                                    seed=self.net_params['seed'])
        # self.xgb_model = xgb.XGBClassifier(params, n_estimators=self.net_params['n_estimators'],seed=self.net_params['seed'])
        from sklearn.multioutput import MultiOutputClassifier
        self.xgb_model = MultiOutputClassifier(xgb_model)
        self.xgb_model.fit(X_train, y_train)

    def fit_trans(self, train_loader, val_loader):
        X_train, y_label = train_loader.data, train_loader.label
        X_test, y_test = val_loader.data, val_loader.label

        # from sklearn.preprocessing import OneHotEncoder
        # y_train = OneHotEncoder(sparse=False).fit_transform(y_label.reshape(-1, 1))
        # y_test = OneHotEncoder(sparse=False).fit_transform(y_test.reshape(-1, 1))

        train = xgb.DMatrix(data=X_train, label=np.int64(y_label))
        val = xgb.DMatrix(data=X_test, label=np.int64(y_test))
        # https://mljar.com/blog/xgboost-save-load-python/
        params = {'objective': 'multi:softprob',
                  'eta': self.eta,
                  'subsample': self.subsample,
                  'max_depth': self.max_depth,
                  'num_class': 4,
                  'eval_metric': 'mlogloss'}

        self.xgb_model = xgb.train(params, train, evals=[(train, "train"), (val, "validation")],
                                   num_boost_round=self.net_params['n_estimators'])
        # self.xgb_model = xgb.train(params, train,
        #                            num_boost_round=self.net_params['n_estimators'])
        # self.xgb_model = xgb.XGBClassifier(params, n_estimators=self.net_params['n_estimators'],seed=self.net_params['seed'])

    def predict_trans(self, data_loader):
        # from sklearn.preprocessing import OneHotEncoder
        # y_train = OneHotEncoder(sparse=False).fit_transform(data_loader.label.reshape(-1, 1))

        test = xgb.DMatrix(data_loader.data, np.int64(data_loader.label))
        preds = self.xgb_model.predict(test, ntree_limit =self.xgb_model.best_ntree_limit)
        yhat_labels = np.argmax(preds, axis=1)
        from sklearn.metrics import confusion_matrix
        print(confusion_matrix(data_loader.label, yhat_labels))
        return preds

    def predict_trans1(self, data_loader):
        X_, y_ = data_loader.data, data_loader.label
        preds = self.xgb_model.predict(X_, ntree_limit=self.xgb_model.best_ntree_limit)
        return preds

    def save_model(self, dir):
        self.xgb_model.save_model((dir + "//model_sklearn.json"))

    def load_model(self, dir):
        model = xgb.XGBClassifier()
        model.load_model((dir + "//model_sklearn.json"))
        return model
