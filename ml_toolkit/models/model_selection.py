from torch import nn
import torch.nn.functional as F
from sklearn import (ensemble, linear_model,
                     naive_bayes, svm, tree)


class ClassMLP(nn.Module):
    def __init__(self, features: int, embedding: int, n_layers=3, dropout=0.2):
        super(ClassMLP, self).__init__()
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
        self.output = nn.Linear(embedding, 2)

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

        X = F.softmax(self.output(X), dim=1)

        return X


def get_sklearn_models():
    models_dict = {
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

    return models_dict
