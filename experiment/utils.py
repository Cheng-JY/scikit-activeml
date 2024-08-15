import hydra
import mlflow
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.datasets import fetch_openml

import torch

from skactiveml.utils import is_labeled


def load_dataset(name, data_dir, random_state=42):
    if name == 'letter':
        return load_dataset_letter(data_dir)
    elif name == 'letter_perf':
        return load_dataset_letter_perf(data_dir)
    elif name == 'dopanim':
        return load_dataset_dopanim(data_dir)
    elif name == 'agnews':
        return load_dataset_agnews(data_dir)


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


def load_dataset_letter_perf(data_dir, random_state=42):

    X = np.load(f'{data_dir}/letter/letter-X.npy').astype(np.float32)
    y_true = np.load(f'{data_dir}/letter/letter-y-true.npy')
    y_3 = np.random.choice(np.unique(y_true), size=len(y_true), replace=True)
    y = np.array([y_true, y_true, y_3], dtype=float).T

    X_train, X_test, y_train, y_test, y_train_true, y_test_true = train_test_split(X, y, y_true, test_size=0.2,
                                                                                   random_state=random_state)
    sc = StandardScaler().fit(X_train)
    X_train = sc.transform(X_train)
    X_test = sc.transform(X_test)
    return X_train, X_test, y_train, y_test, y_train_true, y_test_true


def get_correct_label_ratio(y_partial, y_train_true, missing_label):
    is_lbld = is_labeled(y_partial, missing_label=missing_label)
    number_annotation_annotator = np.sum(is_lbld, axis=0)
    y_true = np.array([y_train_true for _ in range(y_partial.shape[1])]).T
    correct_label = is_lbld * (y_partial == y_true)
    number_correct_label_annotator = np.sum(correct_label, axis=0)
    correct_label_ratio = correct_label.sum() / is_lbld.sum()
    return number_annotation_annotator, number_correct_label_annotator, correct_label_ratio


def load_dataset_dopanim(data_dir):
    X_train = np.load(f'{data_dir}/dopanim/dopanim_x_train.npy')
    X_test = np.load(f'{data_dir}/dopanim/dopanim_x_test.npy')
    y_train = np.load(f'{data_dir}/dopanim/dopanim_y_train.npy')
    y_train_true = np.load(f'{data_dir}/dopanim/dopanim_y_true_train.npy')
    y_test_true = np.load(f'{data_dir}/dopanim/dopanim_y_true_test.npy')

    return X_train, X_test, y_train, None, y_train_true, y_test_true


def load_dataset_agnews(data_dir):
    X_train = np.load(f'{data_dir}/agnews/ag_news_sim_x_train.npy')
    X_test = np.load(f'{data_dir}/agnews/ag_news_sim_x_test.npy')
    y_train = np.load(f'{data_dir}/agnews/ag_news_sim_y_train.npy')
    y_train_true = np.load(f'{data_dir}/agnews/ag_news_sim_y_true_train.npy')
    y_test_true = np.load(f'{data_dir}/agnews/ag_news_sim_y_true_test.npy')

    return X_train, X_test, y_train, None, y_train_true, y_test_true


if __name__ == "__main__":
    data_dir = '/Users/chengjiaying/PycharmProjects/scikit-activeml/experiment/dataset'
    dataset = 'letter'
    X_train, X_test, y_train, y_test, y_train_true, y_test_true = load_dataset(dataset, data_dir=data_dir)

    classes = np.unique(y_train_true)
    n_classes = len(classes)
    n_features = X_train.shape[1]
    n_annotators = y_train.shape[1]
    n_samples = X_train.shape[0]
    test_instances = X_test.shape[0]
    print('n_annotators:', n_annotators)
    print('n_samples:', n_samples)
    print('n_features:', n_features)
    print('n_classes:', n_classes)
    print('test_instances:', n_annotators)


