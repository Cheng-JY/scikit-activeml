import mlflow
import pandas as pd

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