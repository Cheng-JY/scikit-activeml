import mlflow
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_dataset(name, data_dir):
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


def load_dataset_music(data_dir):
    X_train = np.load(f'{data_dir}/music/music-X.npy')
    y_train = np.load(f'{data_dir}/music/music-y.npy')
    y_train_true = np.load(f'{data_dir}/music/music-y-true.npy')
    X_valid = np.load(f'{data_dir}/music/music-X-valid.npy')
    y_valid_true = np.load(f'{data_dir}/music/music-y-true-valid.npy')
    X_test = np.load(f'{data_dir}/music/music-X-test.npy')
    y_test_true = np.load(f'{data_dir}/music/music-y-true-test.npy')

    sc = StandardScaler().fit(X_train)
    X_train = sc.transform(X_train)
    X_valid = sc.transform(X_valid)
    X_test = sc.transform(X_test)

    return X_train, y_train, y_train_true, X_valid, y_valid_true, X_test, y_test_true


def load_dataset_label_me(data_dir):
    X_train = np.load(f'{data_dir}/label-me/label-me-X.npy')
    y_train = np.load(f'{data_dir}/label-me/label-me-y.npy')
    y_train_true = np.load(f'{data_dir}/label-me/label-me-y-true.npy')
    X_valid = np.load(f'{data_dir}/label-me/label-me-X-valid.npy')
    y_valid_true = np.load(f'{data_dir}/label-me/label-me-y-true-valid.npy')
    X_test = np.load(f'{data_dir}/label-me/label-me-X-test.npy')
    y_test_true = np.load(f'{data_dir}/label-me/label-me-y-true-test.npy')

    return X_train, y_train, y_train_true, X_valid, y_valid_true, X_test, y_test_true


def get_experiment_result(uri, experiment_name, path, train_name):
    mlflow.set_tracking_uri(uri=uri)

    experiment = mlflow.get_experiment_by_name(experiment_name)
    df = mlflow.search_runs(experiment_ids=experiment.experiment_id, output_format="pandas")

    if train_name == 'cl':
        csv_df = df[['params.model_name', 'params.batch_size', 'params.max_epochs', 'params.lr',
                 'params.optimizer__weight_decay', 'metrics.train_accuracy', 'metrics.test_accuracy', 'metrics.time']]
    elif train_name == 'label-me':
        print(df.columns)
        csv_df = df[['params.nn_name', 'params.batch_size', 'params.max_epochs', 'params.lr', 'params.seed',
                     'params.optimizer__weight_decay', 'metrics.train_accuracy', 'metrics.test_accuracy',
                     'metrics.time']]
    csv_df.to_csv(path, index=False)


if __name__ == '__main__':
    uri_training = '/Users/chengjiaying/PycharmProjects/scikit-activeml/tutorials/tracking'
    # Label-Me-Training
    # Crowd-Layer-Training
    exp_training = 'Letter-Training'
    path_training = 'training_letter.csv'
    get_experiment_result(uri_training, exp_training, path_training, 'label-me')