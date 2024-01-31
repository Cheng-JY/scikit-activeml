import sys
import time
import warnings
import mlflow
import os

#sys.path.append("/Users/chengjiaying/scikit-activeml/")
sys.path.append("/mnt/stud/home/jcheng/scikit-activeml/")
warnings.filterwarnings("ignore")

import numpy as np

from sklearn.linear_model import LogisticRegression

from skactiveml.classifier import SklearnClassifier
from skactiveml.pool import UncertaintySampling, RandomSampling, DiscriminativeAL, CoreSet, TypiClust, Badge
from skactiveml.utils import call_func, MISSING_LABEL

import argparse
from tqdm import tqdm
import pandas as pd

def parse_argument():
    parser = argparse.ArgumentParser(description='Evaluate model performance')
    parser.add_argument('dataset', type=str, help='name of dataset')
    parser.add_argument('query_strategy', type=str, help='name of query strategy')
    parser.add_argument('batch_size', type=int, help='batch size')
    parser.add_argument('n_cycles', type=int, help='number of cycles')
    parser.add_argument('seed', type=int, help='random seed')
    return parser

def gen_seed(random_state:np.random.RandomState):
    return random_state.randint(0, 2**31)

def gen_random_state(random_state:np.random.RandomState):
    return np.random.RandomState(gen_seed(random_state))

def create_query_strategy(name, random_state):
    return query_strategy_factory_functions[name](random_state)

def load_embedding_dataset(name):
    # X_train = np.load('./embedding_data/flowers102_dinov2B_X_train.npy')
    # y_train_true = np.load('./embedding_data/flowers102_dinov2B_y_train.npy')
    # X_test = np.load('./embedding_data/flowers102_dinov2B_X_test.npy')
    # y_test_true = np.load('./embedding_data/flowers102_dinov2B_y_test.npy')
    X_train = np.load(f'/mnt/stud/home/jcheng/scikit-activeml/tutorials/embedding_data/{name}_dinov2B_X_train.npy')
    y_train_true = np.load(f'/mnt/stud/home/jcheng/scikit-activeml/tutorials/embedding_data/{name}_dinov2B_y_train.npy')
    X_test = np.load(f'/mnt/stud/home/jcheng/scikit-activeml/tutorials/embedding_data/{name}_dinov2B_X_test.npy')
    y_test_true = np.load(f'/mnt/stud/home/jcheng/scikit-activeml/tutorials/embedding_data/{name}_dinov2B_y_test.npy')
    return X_train, y_train_true, X_test, y_test_true

def save_in_csv():
    data = {
            'dataset': [dataset_name],
            'qs': [qs_name],
            'batch_size': [batch_size],
            'n_cycles': [n_cycles],
            'seed': [seed],
            'step': [c],
            'score': [score],
            'time': [end-start]
            }
    df = pd.DataFrame(data=data)
    output_path=f'/mnt/stud/home/jcheng/scikit-activeml/tutorials/csv/{dataset_name}_{n_cycles}_csv.csv'
    #output_path=f'/Users/chengjiaying/scikit-activeml/tutorials/csv/{dataset_name}_{n_cycles}_csv.csv'
    df.to_csv(output_path, mode='a', header=not os.path.exists(output_path))

def save_in_csv_mlflow():
    
    
if __name__ == '__main__':
    parser = parse_argument()
    args = parser.parse_args()
    dataset_name = args.dataset
    qs_name = args.query_strategy
    batch_size = args.batch_size
    n_cycles = args.n_cycles
    seed = args.seed
    master_random_state = np.random.RandomState(seed)

    X_train, y_train_true, X_test, y_test_true = load_embedding_dataset(dataset_name)

    dataset_classes = {
        "cifar10": 10,
        "cifar100": 100,
        "flowers102": 102,
    }
    classes = dataset_classes[dataset_name]

    clf = SklearnClassifier(LogisticRegression(), classes=np.arange(classes), random_state=gen_seed(master_random_state))

    query_strategy_factory_functions = {
        'RandomSampling': lambda random_state: RandomSampling(random_state=gen_seed(random_state)),
        'UncertaintySampling': lambda random_state: UncertaintySampling(random_state=gen_seed(random_state)),
        'DiscriminativeAL': lambda random_state: DiscriminativeAL(random_state=gen_seed(random_state)),
        'CoreSet': lambda random_state: CoreSet(random_state=gen_seed(random_state)),
        'TypiClust': lambda random_state: TypiClust(random_state=gen_seed(random_state)),
        'Badge': lambda random_state: Badge(random_state=gen_seed(random_state))
    }

    qs = create_query_strategy(qs_name, random_state=gen_random_state(master_random_state))

    y_train = np.full(shape=y_train_true.shape, fill_value=MISSING_LABEL)
    clf.fit(X_train, y_train)

    #mlflow.set_tracking_uri(uri="/Users/chengjiaying/scikit-activeml/tutorials/tracking")
    mlflow.set_tracking_uri(uri="file:///mnt/stud/home/jcheng/scikit-activeml/tutorials/tracking")
    exp = mlflow.get_experiment_by_name("Evaluation-Active Learning")
    experiment_id = mlflow.create_experiment(name="Evaluation-Active Learning") if exp is None else exp.experiment_id

    with (mlflow.start_run(experiment_id=experiment_id)):
        for c in tqdm(range(n_cycles), desc=f'{qs_name} for {dataset_name}'):
            start = time.time()
            query_idx = call_func(qs.query, X=X_train, y=y_train, batch_size=batch_size, clf=clf, discriminator=clf)
            end = time.time()
            y_train[query_idx] = y_train_true[query_idx]
            clf.fit(X_train, y_train)
            score = clf.score(X_test, y_test_true)
            mlflow.log_metric(key='score', value=score, step=c)
            mlflow.log_metric(key='time', value=end-start, step=c)
            tags = {
                'dataset': dataset_name,
                'qs': qs_name,
                'batch_size': batch_size,
                'n_cycles': n_cycles,
                'seed': seed,
                'step': c,
            }
            mlflow.set_tags(tags)
            save_in_csv()


