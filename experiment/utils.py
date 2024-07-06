import hydra
import mlflow
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.datasets import fetch_openml

import torch


def load_dataset(name, data_dir,random_state=42):
    if name == 'letter':
        return load_dataset_letter(data_dir)


def load_dataset_letter(data_dir, random_state=42):
    X = np.load(f'{data_dir}/letter/letter-X.npy').astype(np.float32)
    y = np.load(f'{data_dir}/letter/letter-y.npy')
    y_true = np.load(f'{data_dir}/letter/letter-y-true.npy')

    X_train, X_test, y_train, y_test, y_train_true, y_test_true = train_test_split(X, y, y_true, test_size=0.2,
                                                                                   random_state=random_state)
    sc = StandardScaler().fit(X_train)
    X_train = sc.transform(X_train)
    X_test = sc.transform(X_test)
    return X_train, X_test, y_train, y_test, y_train_true, y_test_true

def load_dataset_letter_2(data_dir, random_state=42):
    X, y_true = fetch_openml(data_id=6, cache=True, return_X_y=True)
    X = X.values.astype(np.float32)
    y_true = LabelEncoder().fit_transform(y_true.values)
    y_new = torch.load(f'{data_dir}/letter/letter-annot-mix.pt').numpy()

    train, test = train_test_split(
        np.arange(len(y_true)), test_size=0.2, random_state=0, stratify=y_true
    )
    train, valid = train_test_split(
        train, test_size=500, random_state=0, stratify=y_true[train]
    )

    X_train, y_train_true, X_test, y_test_true = X[train], y_true[train], X[test], y_true[test]
    y_train, y_test = y_new[np.arange(len(train))], y_new[np.arange(len(train) + len(valid), len(y_true))]
    sc = StandardScaler().fit(X_train)
    X_train = sc.transform(X_train)
    X_test = sc.transform(X_test)
    return X_train, X_test, y_train, y_test, y_train_true, y_test_true