import copy
import os
import sys
import time
import warnings

from matplotlib import pyplot as plt

# sys.path.append("/mnt/stud/home/jcheng/scikit-activeml/")
sys.path.append("/Users/chengjiaying/PycharmProjects/scikit-activeml")
warnings.filterwarnings("ignore")

from skactiveml.classifier import SkorchClassifier
from skactiveml.classifier.multiannotator import RegCrowdNetClassifier
from skactiveml.utils import majority_vote
from sklearn.metrics import accuracy_score

from skorch.callbacks import LRScheduler

from torch import nn

from classifier.classifier import MLPModule, TabularClassifierModule
from utils import *


def seed_everything(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)


@hydra.main(config_path="config", config_name="config", version_base="1.1")
def main(cfg):
    running_device = 'local'

    # load dataset
    data_dir = cfg['dataset_file_path'][running_device]
    data_name = cfg['dataset'] if running_device == 'server' else 'dopanim'
    X_train, X_test, y_train, y_test, y_train_true, y_test_true = (
        load_dataset(name=data_name, data_dir=data_dir)
    )

    dataset_classes = np.unique(y_train_true)
    n_classes = len(dataset_classes)
    n_features = X_train.shape[1]
    n_sample = X_train.shape[0]
    n_annotators = y_train.shape[1]
    print(n_classes)
    print(n_features)
    print(n_sample)

    MISSING_LABEL = -1

    seed_everything(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    mlflow.set_tracking_uri(uri='/Users/chengjiaying/PycharmProjects/scikit-activeml/experiment/tracking')
    exp = mlflow.get_experiment_by_name(name='Dopanim-Training')
    experiment_id = mlflow.create_experiment(name='Dopanim-Training') if exp is None else exp.experiment_id

    with mlflow.start_run(experiment_id=experiment_id) as active_run:
        hyper_parameter = {
            'max_epochs': 50,
            'batch_size': 32,
            'lr': 0.001,
            'optimizer__weight_decay': 0.0,
            'random_state': 1,
        }
        lr_scheduler = LRScheduler(policy="CosineAnnealingLR", T_max=hyper_parameter['max_epochs'])

        y_agg = majority_vote(y_train, classes=dataset_classes, missing_label=MISSING_LABEL,
                              random_state=hyper_parameter['random_state'])
        accuracy = accuracy_score(y_agg, y_train_true)

        net = SkorchClassifier(
            TabularClassifierModule,
            module__n_classes=n_classes,
            module__n_features=n_features,
            module__dropout=0.5,
            classes=dataset_classes,
            missing_label=MISSING_LABEL,
            cost_matrix=None,
            criterion=nn.CrossEntropyLoss(),
            train_split=None,
            verbose=False,
            optimizer=torch.optim.RAdam,
            device=device,
            callbacks=[lr_scheduler],
            iterator_train__drop_last=True,
            iterator_train__shuffle=True,
            **hyper_parameter,
        )

        mlflow.log_params(hyper_parameter)
        y_agg = majority_vote(y_train, classes=dataset_classes, missing_label=MISSING_LABEL,
                              random_state=hyper_parameter['random_state'])

        start = time.time()
        net.fit(X_train, y_agg)
        end = time.time()

        print(net.predict(X_train[40:50]))
        print(y_train_true[40:50])
        train_accuracy = net.score(X_train, y_train_true)
        test_accuracy = net.score(X_test, y_test_true)

        metrics = {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'time': end - start,
        }
        mlflow.log_metrics(metrics)
        print(metrics)

        history = net.history
        train_loss = history[:, 'train_loss']

        plt.plot(train_loss)
        plt.title(f'lr: {hyper_parameter["lr"]}; weight decay: {hyper_parameter["optimizer__weight_decay"]}\n'
                  f'batch_size: {hyper_parameter["batch_size"]} \n'
                  f'train: {metrics["train_accuracy"]}; test: {metrics["test_accuracy"]}')
        plt.show()

        loss = {'train_loss': train_loss}
        df = pd.DataFrame.from_dict(data=loss)
        outpath = active_run.info.artifact_uri
        outpath = os.path.join(outpath, "result.csv")
        df.to_csv(outpath, index=False)
        return


if __name__ == '__main__':
    main()
