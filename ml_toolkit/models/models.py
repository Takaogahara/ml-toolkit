from torch import nn
import torch.nn.functional as F
from sklearn import (ensemble, linear_model,
                     naive_bayes, svm, tree,
                     neighbors)


SK_MODELS = {
    # Ensemble - Classifier
    "adaboostclassifier": ensemble.AdaBoostClassifier,
    "baggingclassifier": ensemble.BaggingClassifier,
    "extratreesclassifier": ensemble.ExtraTreesClassifier,
    "gradientboostingclassifier": ensemble.GradientBoostingClassifier,
    "randomforestclassifier": ensemble.RandomForestClassifier,

    # GLM - Classifier
    "sgdclassifier": linear_model.SGDClassifier,

    # Navies Bayes - Classifier
    "bernoullinb": naive_bayes.BernoulliNB,
    "gaussiannb": naive_bayes.GaussianNB,

    # SVM - Classifier
    "svc": svm.SVC,

    # Trees - Classifier
    "decisiontreeclassifier": tree.DecisionTreeClassifier,
    "extratreeclassifier": tree.ExtraTreeClassifier,

    # Neighbors - Classifier
    "kneighborsclassifier": neighbors.KNeighborsClassifier,

    # Ensemble - Regressor
    "adaboostregressor": ensemble.AdaBoostRegressor,
    "baggingregressor": ensemble.BaggingRegressor,
    "extratreesregressor": ensemble.ExtraTreesRegressor,
    "gradientboostingregressor": ensemble.GradientBoostingRegressor,
    "randomforestregressor": ensemble.RandomForestRegressor,

    # GLM - Regressor
    "sgdregressor": linear_model.SGDRegressor,

    # SVM - Regressor
    "svr": svm.SVR,

    # Trees - Regressor
    "decisiontreeregressor": tree.DecisionTreeRegressor,
    "extratreeregressor": tree.ExtraTreeRegressor}


class BasicMLP(nn.Module):
    def __init__(self, features: int, embedding: int, n_layers=3, dropout=0.2):
        super(BasicMLP, self).__init__()
        self.n_layers = n_layers

        if self.n_layers < 2:
            raise RuntimeError("n_layers must be >= 2")

        self.dense_layers = nn.ModuleList([])
        self.dropout_layers = nn.ModuleList([])

        # Dense blocks
        self.dense0 = nn.Linear(features, embedding)
        for i in range(self.n_layers - 2):
            self.dense_layers.append(
                nn.Linear(embedding, embedding))

        self.output = nn.Linear(embedding, 1)

        # Dropout Blocks
        self.dropout_0 = nn.Dropout(dropout)
        for i in range(self.n_layers - 2):
            self.dropout_layers.append(nn.Dropout(dropout))

    def forward(self, X, **kwargs):
        X = F.relu(self.dense0(X))
        X = self.dropout_0(X)

        for i in range(self.n_layers - 2):
            X = F.relu(self.dense_layers[i](X))
            X = self.dropout_layers[i](X)

        X = self.output(X)

        return X
