import matplotlib as mlp
import matplotlib.pyplot as plt
import numpy as np

from copy import deepcopy

from skactiveml.classifier import SkorchClassifier
from skactiveml.classifier.multiannotator import CrowdLayerClassifier
from skactiveml.pool import RandomSampling
from skactiveml.pool.multiannotator import SingleAnnotatorWrapper
from skactiveml.utils import majority_vote
from skorch.callbacks import LRScheduler

import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm


class ClassifierModule(nn.Module):
    def __init__(self, n_classes, n_features, dropout):
        super(ClassifierModule, self).__init__()
        n_hidden_neurons_1 = 256
        n_hidden_neurons_2 = 128
        self.embed_X_block = nn.Sequential(
            nn.Linear(in_features=n_features, out_features=n_hidden_neurons_1),
            nn.BatchNorm1d(num_features=n_hidden_neurons_1),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=n_hidden_neurons_1, out_features=n_hidden_neurons_2),
            nn.BatchNorm1d(num_features=n_hidden_neurons_2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )
        self.mlp = nn.Linear(in_features=n_hidden_neurons_2, out_features=n_classes)

    def forward(self, x):
        embed_x = self.embed_X_block(x)
        logit_class = self.mlp(embed_x)

        return logit_class


def load_data_set_label_me(data_dir):
    ds = {}

    X_train = np.load(f'{data_dir}/label-me-X.npy')
    y_train = np.load(f'{data_dir}/label-me-y.npy')
    y_train_true = np.load(f'{data_dir}/label-me-y-true.npy')
    X_valid = np.load(f'{data_dir}/label-me-X-valid.npy')
    y_valid_true = np.load(f'{data_dir}/label-me-y-true-valid.npy')
    X_test = np.load(f'{data_dir}/label-me-X-test.npy')
    y_test_true = np.load(f'{data_dir}/label-me-y-true-test.npy')

    ds['X_train'] = X_train
    ds['y_train'] = y_train
    ds['y_train_true'] = y_train_true
    ds['X_valid'] = X_valid
    ds['y_valid_true'] = y_valid_true
    ds['X_test'] = X_test
    ds['y_test_true'] = y_test_true

    return ds


if __name__ == '__main__':
    MISSING_LABEL = -1
    RANDOM_STATE = 0

    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)
    torch.cuda.manual_seed(RANDOM_STATE)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load dataset
    data_dir = f'./data/label-me'
    ds = load_data_set_label_me(data_dir)

    classes = np.unique(ds['y_train_true'])
    n_classes = len(classes)
    X_train = ds['X_train'].astype(np.float32)
    y_train = ds['y_train']
    n_features = X_train.shape[1]
    n_annotators = y_train.shape[1]
    n_samples = X_train.shape[0]

    # neural network
    hyper_parameter = {
        'max_epochs': 50,
        'batch_size': 64,
        'lr': 0.01,
        'optimizer__weight_decay': 0.0001,
    }
    lr_scheduler = LRScheduler(policy='CosineAnnealingLR', T_max=hyper_parameter['max_epochs'])

    net_mv = SkorchClassifier(
        ClassifierModule,
        module__n_classes=n_classes,
        module__n_features=n_features,
        module__dropout=0.5,
        classes=classes,
        missing_label=MISSING_LABEL,
        cost_matrix=None,
        random_state=1,
        criterion=nn.CrossEntropyLoss(),
        train_split=None,
        verbose=False,
        optimizer=torch.optim.RAdam,
        device=device,
        callbacks=[lr_scheduler],
        **hyper_parameter
    )

    # active learning
    sa_qs = RandomSampling(random_state=0, missing_label=MISSING_LABEL)
    ma_qs = SingleAnnotatorWrapper(sa_qs, random_state=0, missing_label=MISSING_LABEL)

    idx = lambda A: (A[:, 0], A[:, 1])

    n_cycle = 20

    # the already observed labels for each sample and annotator
    y = np.full(shape=(n_samples, n_annotators), fill_value=MISSING_LABEL, dtype=np.int32)
    y_init = np.full_like(ds['y_train_true'], fill_value=MISSING_LABEL, dtype=np.int32)

    query_idx = sa_qs.query(X_train, y_init, batch_size=64)
    y[query_idx] = y_train[query_idx]

    y_mv = majority_vote(y, random_state=RANDOM_STATE, missing_label=MISSING_LABEL)
    net_mv.fit(X_train, ds['y_train_true'])
    # print(net_mv.score(ds['X_test'], ds['y_test_true']))

    # for c in range(n_cycle):
    #     query_idx = ma_qs.query(X_train, y, batch_size=64, n_annotators_per_sample=2)
    #     y[idx(query_idx)] = y_train[idx(query_idx)]
    #     y_mv = majority_vote(y, random_state=RANDOM_STATE, missing_label=MISSING_LABEL)
    #     net_mv.fit(X_train, y_mv)
    #     print('cycle ', c, net_mv.score(ds['X_test'], ds['y_test_true']))

