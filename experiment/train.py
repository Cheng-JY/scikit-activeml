import os
import sys
import time

from matplotlib import pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils import check_random_state
from skorch.callbacks import LRScheduler

import numpy as np
import pandas as pd

import torch
from torch import nn

from sklearn.metrics import accuracy_score
from skactiveml.classifier.multiannotator import RegCrowdNetClassifier

import mlflow
from utils import *

sys.path.append('../../..')


# load dataset music and label-me


def seed_everything(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)


class GT_Embed_Module(nn.Module):
    def __init__(self, n_features, dropout):
        super(GT_Embed_Module, self).__init__()
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

    def forward(self, x):
        embed_x = self.embed_X_block(x)

        return embed_x


class GT_Output_Module(nn.Module):
    def __init__(self, n_classes):
        super(GT_Output_Module, self).__init__()
        n_hidden_neurons_2 = 128
        self.mlp = nn.Linear(in_features=n_hidden_neurons_2, out_features=n_classes)

    def forward(self, x):
        logit_class = self.mlp(x)

        return logit_class


if __name__ == '__main__':
    seed = 0
    MISSING_LABEL = -1

    data_dir = "/Users/chengjiaying/PycharmProjects/scikit-activeml/experiment/dataset"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed_everything(seed)
    X_train, y_train, y_train_true, X_valid, y_valid_true, X_test, y_test_true = load_dataset_music(data_dir)

    dataset_classes = np.unique(y_test_true)
    n_classes = len(dataset_classes)
    n_features = X_train.shape[1]
    n_annotators = y_train.shape[1]

    mlflow.set_tracking_uri(uri='/Users/chengjiaying/PycharmProjects/scikit-activeml/experiment/tracking')
    exp = mlflow.get_experiment_by_name(name='Letter-Training')
    experiment_id = mlflow.create_experiment(name='Letter-Training') if exp is None else exp.experiment_id

    with mlflow.start_run(experiment_id=experiment_id) as active_run:
        hyper_dict = {
            'max_epochs': 50,
            'batch_size': 64,
            'lr': 0.01,
            'optimizer__weight_decay': 0.0,
        }
        lr_scheduler = LRScheduler(policy="CosineAnnealingLR", T_max=hyper_dict['max_epochs'])

        regularization = "geo-reg-w"
        gt_net = GT_Embed_Module(n_features=n_features, dropout=0.5)
        output_net = GT_Output_Module(n_classes=n_classes)
        net = RegCrowdNetClassifier(
            module__gt_embed_x=gt_net,
            module__gt_output=output_net,
            n_classes=n_classes,
            n_annotators=n_annotators,
            classes=dataset_classes,
            missing_label=MISSING_LABEL,
            cost_matrix=None,
            random_state=seed,
            train_split=None,
            verbose=False,
            optimizer=torch.optim.RAdam,
            device=device,
            callbacks=[lr_scheduler],
            lmbda="auto",
            regularization=regularization,
            **hyper_dict
        )
        hyper_dict['seed'] = seed
        mlflow.log_params(hyper_dict)

        start = time.time()
        net.fit(X_train, y_train)
        end = time.time()

        train_accuracy = net.score(X_train, y_train_true)

        p_pred = net.predict_proba(X_test)
        test_accuracy = net.score(X_test, y_test_true)
        metrics = {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'time': end-start,
        }
        mlflow.log_metrics(metrics)
        print(metrics)
        print(net.ap_confs)
        annot_pref = net.predict_annotator_perf(False)
        print(annot_pref)
        confusion_pref = net.predict_annotator_perf(True)
        print(confusion_pref.shape)

        history = net.history
        train_loss = history[:, 'train_loss']

        plt.plot(train_loss)
        plt.title(f'lr: {hyper_dict["lr"]}; weight decay: {hyper_dict["optimizer__weight_decay"]}\n'
                  f'train: {metrics["train_accuracy"]}; test: {metrics["test_accuracy"]}')
        plt.show()

        loss = {'train_loss': train_loss}
        df = pd.DataFrame.from_dict(data=loss)
        outpath = active_run.info.artifact_uri
        outpath = os.path.join(outpath, "result.csv")
        df.to_csv(outpath, index=False)