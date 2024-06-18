import os
import sys
import time

from matplotlib import pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils import check_random_state
from skorch.callbacks import LRScheduler

from skactiveml.utils import majority_vote

sys.path.append('..')

import numpy as np
import pandas as pd

import torch
from torch import nn

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from skactiveml.classifier import SkorchClassifier
from skactiveml.classifier.multiannotator import CrowdLayerClassifier

import mlflow

def introduce_missing_annotations(y, percentage, missing_label=-1, random_state=None):
    """
    Randomly replace a specified proportion of annotations with `missing_label`.

    Parameters
    ----------
    y : numpy.ndarray of shape (n_samples, n_annotators)
        Annotation of each annotator for each sample.
    percentage : float in [0, 1]
        Percentage of annotations that should be replaced with `missing_label`.
    missing_label : scalar or string or np.nan or None, optional (default=-1)
        Value to represent a missing label.
    random_state : int or None, optional (default=None)
        Determines seed for reproducible results.


    Returns
    -------
    y_missing : numpy.ndarray of shape (n_samples, n_annotators)
        Annotations with randomly introduced missing labels.
    """
    y_missing = np.array(y.copy(), dtype=float)

    if percentage >= 1:
        no_missing_annotations = (y.shape[0] * y.shape[1]) - (y.shape[0] * percentage)
    else:
        no_missing_annotations = round((y.shape[0] * y.shape[1]) * percentage)

    is_missing = np.zeros(y.shape[0] * y.shape[1], dtype=bool)
    is_missing[:no_missing_annotations] = 1

    random_state = check_random_state(random_state)
    random_state.shuffle(is_missing)

    is_missing = is_missing.reshape(y.shape)

    y_missing[is_missing] = missing_label

    return y_missing

def load_dataset_letter(random_state=42):
    data_dir = 'dataset/letter'
    # X = np.load(f'{data_dir}/letter-X.npy')
    # y = np.load(f'{data_dir}/letter-y.npy')
    X, y_true = fetch_openml(data_id=6, cache=True, return_X_y=True)
    # y_true = np.load(f'{data_dir}/letter-y-true.npy')
    X = X.values.astype(np.float32)
    y_true = LabelEncoder().fit_transform(y_true.values)
    y_new = torch.load(f'{data_dir}/letter-annot-mix.pt').numpy()
    # print(y_new.shape)

    # X_train, X_test, y_train, y_test, y_train_true, y_test_true = train_test_split(X, y, y_true, test_size=0.2, random_state=random_state)
    # X_train, X_test, y_train, y_test, y_train_true, y_test_true = train_test_split(X, y_new, y_true, test_size=0.2,
    #                                                                                random_state=random_state)

    train, test = train_test_split(
        np.arange(len(y_true)), test_size=0.2, random_state=0, stratify=y_true
    )
    train, valid = train_test_split(
        train, test_size=500, random_state=0, stratify=y_true[train]
    )

    X_train, y_train_true, X_test, y_test_true = X[train], y_true[train], X[test], y_true[test]
    y_train, y_test = y_new[np.arange(len(train))], y_new[np.arange(len(train)+len(valid), len(y_true))]
    sc = StandardScaler().fit(X_train)
    X_train = sc.transform(X_train)
    X_test = sc.transform(X_test)
    return X_train, X_test, y_train, y_test, y_train_true, y_test_true


def seed_everything(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)


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


if __name__ == '__main__':
    seed = 0
    MISSING_LABEL = -1

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed_everything(seed)
    X_train, X_test, y_train, y_test, y_train_true, y_test_true = load_dataset_letter(seed)
    y_train = introduce_missing_annotations(y_train, percentage=0.8, missing_label=MISSING_LABEL)

    dataset_classes = np.unique(y_test_true)
    n_classes = len(dataset_classes)
    n_features = X_train.shape[1]
    n_annotators = y_train.shape[1]

    mlflow.set_tracking_uri(uri='/Users/chengjiaying/PycharmProjects/scikit-activeml/tutorials/tracking')
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

        nn_name = 'cl'
        if nn_name == 'cl':
            gt_net = ClassifierModule(n_classes=n_classes, n_features=n_features, dropout=0.5)
            net = CrowdLayerClassifier(
                module__n_classes=n_classes,
                module__n_annotators=n_annotators,
                module__gt_net=gt_net,
                classes=dataset_classes,
                missing_label=MISSING_LABEL,
                cost_matrix=None,
                random_state=seed,
                train_split=None,
                verbose=False,
                optimizer=torch.optim.RAdam,
                device=device,
                callbacks=[lr_scheduler],
                **hyper_dict
            )
        elif nn_name == 'ub' or nn_name == 'lb':
            net = SkorchClassifier(
                ClassifierModule,
                module__n_classes=n_classes,
                module__n_features=n_features,
                module__dropout=0.5,
                classes=dataset_classes,
                missing_label=MISSING_LABEL,
                cost_matrix=None,
                random_state=1,
                criterion=nn.CrossEntropyLoss(),
                train_split=None,
                verbose=False,
                optimizer=torch.optim.RAdam,
                device=device,
                callbacks=[lr_scheduler],
                **hyper_dict
            )
            if nn_name == 'ub':
                y_train = y_train_true
            elif nn_name == 'lb':
                y_train = majority_vote(y_train, classes=dataset_classes, missing_label=-1)
                print(accuracy_score(y_train, y_train_true))

        hyper_dict['nn_name'] = nn_name
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