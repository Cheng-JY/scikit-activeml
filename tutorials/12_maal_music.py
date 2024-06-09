import matplotlib as mlp
import matplotlib.pyplot as plt
import numpy as np

from copy import deepcopy

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from skactiveml.classifier import SkorchClassifier
from skactiveml.classifier.multiannotator import CrowdLayerClassifier
from skactiveml.pool import RandomSampling
from skactiveml.pool.multiannotator import SingleAnnotatorWrapper
from skactiveml.utils import majority_vote, is_labeled, is_unlabeled
from skorch.callbacks import LRScheduler

import torch
from torch import nn


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


def load_dataset_music():
    data_dir = 'dataset/music'

    X_train = np.load(f'{data_dir}/music-X.npy')
    y_train = np.load(f'{data_dir}/music-y.npy')
    y_train_true = np.load(f'{data_dir}/music-y-true.npy')
    X_valid = np.load(f'{data_dir}/music-X-valid.npy')
    y_valid_true = np.load(f'{data_dir}/music-y-true-valid.npy')
    X_test = np.load(f'{data_dir}/music-X-test.npy')
    y_test_true = np.load(f'{data_dir}/music-y-true-test.npy')

    sc = StandardScaler().fit(X_train)
    X_train = sc.transform(X_train)
    X_valid = sc.transform(X_valid)
    X_test = sc.transform(X_test)

    return X_train, y_train, y_train_true, X_valid, y_valid_true, X_test, y_test_true


def seed_everything(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)


if __name__ == '__main__':
    MISSING_LABEL = -1
    RANDOM_STATE = 0

    seed_everything(RANDOM_STATE)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load dataset
    X_train, y_train, y_train_true, X_valid, y_valid_true, X_test, y_test_true = load_dataset_music()

    print(is_labeled(y_train, missing_label=MISSING_LABEL).sum())
    classes = np.unique(y_train_true)
    n_classes = len(classes)
    n_features = X_train.shape[1]
    n_annotators = y_train.shape[1]
    n_samples = X_train.shape[0]

    # performance
    accuracies = []

    # Generate model according to config
    hyper_parameter = {
        'max_epochs': 50,
        'batch_size': 64,
        'lr': 0.01,
        'optimizer__weight_decay': 0.0,
    }
    lr_scheduler = LRScheduler(policy='CosineAnnealingLR', T_max=hyper_parameter['max_epochs'])

    # randomly add missing labels
    y_partial = np.full_like(y_train, fill_value=MISSING_LABEL)
    initial_label_size = 10

    for a_idx in range(n_annotators):
        random_state = np.random.RandomState(a_idx)
        is_lbld_a = is_labeled(y_train[:,a_idx], missing_label=MISSING_LABEL)
        p_a = is_lbld_a / is_lbld_a.sum()
        initial_label_size = min(initial_label_size, is_lbld_a.sum())
        selected_idx_a = random_state.choice(np.arange(n_samples), size=initial_label_size, p=p_a, replace=False)
        y_partial[selected_idx_a, a_idx] = y_train[selected_idx_a, a_idx]

    print(is_labeled(y_partial, missing_label=MISSING_LABEL).sum())

    # Create query strategy
    sa_qs = RandomSampling(random_state=RANDOM_STATE, missing_label=MISSING_LABEL)
    ma_qs = SingleAnnotatorWrapper(sa_qs, random_state=RANDOM_STATE, missing_label=MISSING_LABEL)
    candidate_indices = np.arange(n_samples)

    # Function to be able to index via an array of indices
    idx = lambda A: (A[:, 0], A[:, 1])

    # Fallback for random annotator selection
    A_random = np.ones_like(y_partial)

    n_al_cycle = 25
    al_batch_size = 16
    nn_name = ('cl')

    for c in range(n_al_cycle+1):
        if c > 0:
            # set the performance
            if nn_name in ['mv', 'cl-random']:
                A_perf = A_random
            else:
                A_perf = net.predict_annotator_perf(X_train)
            y_query = np.copy(y_partial)
            is_ulbld_query = np.copy(is_ulbld)
            is_candidate = is_ulbld_query.all(axis=-1)
            candidates = candidate_indices[is_candidate]
            available_annotator = is_labeled(y_train, missing_label=MISSING_LABEL)
            query_indices = ma_qs.query(
                X=X_train,
                y=y_partial,
                candidates=candidates,
                A_perf=A_perf[candidates],
                batch_size=al_batch_size,
                annotators=available_annotator[candidates],
                n_annotators_per_sample=1,
            )
            y_partial[idx(query_indices)] = y_train[idx(query_indices)]
            print(y_partial[idx(query_indices)])

        if nn_name in ['mv']:
            # Generate model
            net = SkorchClassifier(
                    ClassifierModule,
                    module__n_classes=n_classes,
                    module__n_features=n_features,
                    module__dropout=0.5,
                    classes=classes,
                    missing_label=MISSING_LABEL,
                    cost_matrix=None,
                    random_state=RANDOM_STATE+c,
                    criterion=nn.CrossEntropyLoss(),
                    train_split=None,
                    verbose=False,
                    optimizer=torch.optim.RAdam,
                    device=device,
                    callbacks=[lr_scheduler],
                    **hyper_parameter
                )
            y_agg = majority_vote(y_partial, classes=classes, missing_label=MISSING_LABEL, random_state=RANDOM_STATE + c)
            net.fit(X_train, y_agg)
        else:
            gt_net = ClassifierModule(n_classes=n_classes, n_features=n_features, dropout=0.5)
            net = CrowdLayerClassifier(
                module__n_classes=n_classes,
                module__n_annotators=n_annotators,
                module__gt_net=gt_net,
                classes=classes,
                missing_label=MISSING_LABEL,
                cost_matrix=None,
                random_state=RANDOM_STATE+c,
                train_split=None,
                verbose=False,
                optimizer=torch.optim.RAdam,
                device=device,
                callbacks=[lr_scheduler],
                **hyper_parameter,
            )
            net.fit(X_train, y_partial)

        accuracy = net.score(X_test, y_test_true)
        accuracies.append(accuracy)
        print(c, accuracy)
        is_ulbld = is_unlabeled(y_partial, missing_label=MISSING_LABEL)

    print(is_labeled(y_partial, missing_label=MISSING_LABEL).sum())
    plt.plot(accuracies)
    plt.title(f'{nn_name} + music + majority-voting + random-sampling')
    plt.show()