import argparse
import copy
import os
import sys
import warnings

sys.path.append("/mnt/stud/home/jcheng/scikit-activeml/")
# sys.path.append("/Users/chengjiaying/PycharmProjects/scikit-activeml")
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import hydra
import pandas as pd

from copy import deepcopy

from skactiveml.classifier import SkorchClassifier
from skactiveml.pool.multiannotator import SingleAnnotatorWrapper
from skactiveml.utils import majority_vote, is_labeled, is_unlabeled

from skorch.callbacks import LRScheduler

import torch
from torch import nn

from classifier.classifier import TabularClassifierModule
from utils import load_dataset
from query_utils import create_instance_query_strategy, get_annotator_performance, gen_random_state


def seed_everything(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_correct_label_ratio(y_partial, y_train_true, missing_label):
    is_lbld = is_labeled(y_partial, missing_label=missing_label)
    y_true = np.array([y_train_true for _ in range(y_partial.shape[1])]).T
    correct_label_ration = is_lbld * (y_partial == y_true)
    correct_label_ration = correct_label_ration.sum() / is_lbld.sum()
    return correct_label_ration


@hydra.main(config_path="config", config_name="config", version_base="1.1")
def main(cfg):
    print(cfg)

    experiment_params = {
        'dataset_name': cfg['dataset'],
        'instance_query_strategy': cfg['instance_query_strategy'],
        'annotator_query_strategy': cfg['annotator_query_strategy'],
        'batch_size': cfg['batch_size'],
        'n_annotators_per_sample': cfg['n_annotator_per_instance'],
        'n_cycles': cfg['n_cycles'],
        'seed': cfg['seed'],
    }
    master_random_state = np.random.RandomState(experiment_params['seed'])

    experiment_params = {
        'dataset_name': 'letter',
        'instance_query_strategy': cfg['instance_query_strategy'],
        'annotator_query_strategy': cfg['annotator_query_strategy'],
        'batch_size': cfg['batch_size'],
        'n_annotators_per_sample': cfg['n_annotator_per_instance'],
        'n_cycles': cfg['n_cycles'],
        'seed': cfg['seed'],
    }
    master_random_state = np.random.RandomState(experiment_params['seed'])

    metric_dict = {
        'step': [],
        'misclassification': [],
        'erorr_annotation_rate': [],
    }

    MISSING_LABEL = -1

    seed_everything(experiment_params['seed'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load dataset
    data_dir = cfg['dataset_file_path']['server']
    X_train, X_test, y_train, y_test, y_train_true, y_test_true = (
        load_dataset(name=experiment_params['dataset_name'], data_dir=data_dir))

    classes = np.unique(y_train_true)
    n_classes = len(classes)
    n_features = X_train.shape[1]
    n_annotators = y_train.shape[1]
    n_samples = X_train.shape[0]

    # Generate model according to config
    hyper_parameter = cfg['hyper_parameter']
    lr_scheduler = LRScheduler(policy='CosineAnnealingLR', T_max=hyper_parameter['max_epochs'])

    # randomly add missing labels
    y_partial = np.full_like(y_train, fill_value=MISSING_LABEL)
    initial_label_size = 100

    for a_idx in range(n_annotators):
        random_state = np.random.RandomState(a_idx)
        is_lbld_a = is_labeled(y_train[:, a_idx], missing_label=MISSING_LABEL)
        p_a = is_lbld_a / is_lbld_a.sum()
        initial_label_size = min(initial_label_size, is_lbld_a.sum())
        selected_idx_a = random_state.choice(np.arange(n_samples), size=initial_label_size, p=p_a, replace=False)
        y_partial[selected_idx_a, a_idx] = y_train[selected_idx_a, a_idx]

    correct_label_ratio = get_correct_label_ratio(y_partial, y_train_true, MISSING_LABEL)
    metric_dict['erorr_annotation_rate'].append(1 - correct_label_ratio)

    # Create query strategy
    sa_qs = create_instance_query_strategy(experiment_params['instance_query_strategy'], random_state=gen_random_state(master_random_state), missing_label=MISSING_LABEL)
    ma_qs = SingleAnnotatorWrapper(sa_qs, random_state=gen_random_state(master_random_state), missing_label=MISSING_LABEL)

    candidate_indices = np.arange(n_samples)

    # Function to be able to index via an array of indices
    idx = lambda A: (A[:, 0], A[:, 1])

    # Fallback for random annotator selection
    A = get_annotator_performance(experiment_params['annotator_query_strategy'], y_partial.shape)

    ml_flow_tracking = cfg['ml_flow_tracking']
    mlflow.set_tracking_uri(uri=ml_flow_tracking["tracking_file_path_server"])
    exp = mlflow.get_experiment_by_name(name=ml_flow_tracking["experiment_name"])
    experiment_id = mlflow.create_experiment(name=ml_flow_tracking["experiment_name"]) \
        if exp is None else exp.experiment_id

    with (mlflow.start_run(experiment_id=experiment_id) as active_run):
        mlflow.log_params(experiment_params)

        for c in range(experiment_params['n_cycles'] + 1):
            if c > 0:
                if experiment_params['annotator_query_strategy'] == 'random':
                    A_perf = A
                elif experiment_params['annotator_query_strategy'] == 'round-robin':
                    A_perf = copy.deepcopy(A)
                    res_anno = ((c-1) * experiment_params['n_annotators_per_sample']) % n_annotators
                    A_perf[:, res_anno: res_anno+experiment_params['n_annotators_per_sample']] = 1
                is_ulbld_query = np.copy(is_ulbld)
                is_candidate = is_ulbld_query.all(axis=-1)
                candidates = candidate_indices[is_candidate]
                query_indices = ma_qs.query(
                    X=X_train,
                    y=y_partial,
                    candidates=candidates,
                    A_perf=A_perf[candidates],
                    batch_size=experiment_params['batch_size'],
                    n_annotators_per_sample=experiment_params['n_annotators_per_sample'],
                )
                y_partial[idx(query_indices)] = y_train[idx(query_indices)]
                correct_label_ratio = get_correct_label_ratio(y_partial, y_train_true, MISSING_LABEL)
                metric_dict['erorr_annotation_rate'].append(correct_label_ratio)

            net = SkorchClassifier(
                TabularClassifierModule,
                module__n_classes=n_classes,
                module__n_features=n_features,
                module__dropout=0.5,
                classes=classes,
                missing_label=MISSING_LABEL,
                cost_matrix=None,
                random_state=experiment_params['seed'] + c,
                criterion=nn.CrossEntropyLoss(),
                train_split=None,
                verbose=False,
                optimizer=torch.optim.RAdam,
                device=device,
                callbacks=[lr_scheduler],
                **hyper_parameter
            )

            y_agg = majority_vote(y_partial, classes=classes, missing_label=MISSING_LABEL,
                                  random_state=experiment_params['seed'] + c)
            net.fit(X_train, y_agg)

            accuracy = net.score(X_test, y_test_true)
            metric_dict['misclassification'].append(1 - accuracy)
            metric_dict['step'].append(c)

            print(c, accuracy)
            is_ulbld = is_unlabeled(y_partial, missing_label=MISSING_LABEL)
        
        df = pd.DataFrame.from_dict(data=metric_dict)
        outpath = active_run.info.artifact_uri
        print(active_run.info.artifact_uri)
        outpath = os.path.join(outpath, 'result.csv')
        df.to_csv(outpath, index=False)
        return


if __name__ == '__main__':
    main()





