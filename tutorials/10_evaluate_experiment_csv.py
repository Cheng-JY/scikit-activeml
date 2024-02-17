import pandas as pd
import matplotlib as mlp
import matplotlib.pyplot as plt
import warnings
import numpy as np

import argparse

mlp.rcParams["figure.facecolor"] = "white"
warnings.filterwarnings("ignore")


def parse_argument():
    parser = argparse.ArgumentParser(description='Evaluate model performance')
    parser.add_argument('dataset', type=str, help='name of dataset')
    parser.add_argument('n_cycles', type=int, help='n_cycles')
    return parser


def evaluate_experiment_csv_score(dataset_name):
    # load csx
    # input_path = f'/Users/chengjiaying/scikit-activeml/tutorials/csv/{dataset_name}_csv.csv'
    input_path = f'/mnt/stud/home/jcheng/scikit-activeml/tutorials/csv_3/{dataset_name}_{n_cycles}_csv.csv'
    dataframe = pd.read_csv(input_path, index_col=0, on_bad_lines='skip')
    dataframe = dataframe.dropna()

    query_strategy_names = dataframe['qs'].unique()
    print(query_strategy_names)

    result_score = dataframe.groupby(['qs', 'batch_size', 'n_cycles', 'step'])['score'].agg(['mean', 'std']).set_axis(
        ['s_mean', 's_std'], axis=1)

    for qs_name in query_strategy_names:
        qs_result_s = result_score.loc[qs_name]
        qs_result_s_mean = qs_result_s['s_mean'].to_numpy()
        qs_result_s_std = qs_result_s['s_std'].to_numpy()

        plt.errorbar(np.arange(16, (len(qs_result_s_mean)+1)*16, 16), qs_result_s_mean, qs_result_s_std,
                     label=f"({np.mean(qs_result_s_mean):.4f}) {qs_name}", alpha=0.5)

    plt.legend(loc='lower right')
    plt.xlabel('# Labels queried')
    plt.ylabel("Accuracy")
    # output_path = f'{dataset_name}_score.png'
    output_path = f'/mnt/stud/home/jcheng/scikit-activeml/tutorials/result/{dataset_name}_score.pdf'
    plt.savefig(output_path)


def evaluate_experiment_csv_time(dataset_name):
    # input_path = f'/Users/chengjiaying/scikit-activeml/tutorials/csv/{dataset_name}_csv.csv'
    input_path = f'/mnt/stud/home/jcheng/scikit-activeml/tutorials/csv_3/{dataset_name}_{n_cycles}_csv.csv'
    dataframe = pd.read_csv(input_path, index_col=0, on_bad_lines='skip')
    dataframe = dataframe.dropna()

    query_strategy_names = dataframe['qs'].unique()

    result_time = dataframe.groupby(['qs', 'batch_size', 'n_cycles', 'step'])['time'].agg(['mean', 'std']).set_axis(
        ['t_mean', 't_std'], axis=1)

    for qs_name in query_strategy_names:
        qs_result_t = result_time.loc[qs_name]
        qs_result_t_mean = qs_result_t['t_mean'].to_numpy()
        qs_result_t_std = qs_result_t['t_std'].to_numpy()

        plt.errorbar(np.arange(16, (len(qs_result_t_mean)+1)*16, 16), qs_result_t_mean, qs_result_t_std,
                     label=f"({np.mean(qs_result_t_mean):.4f}) {qs_name}", alpha=0.3)

    plt.title(dataset_name)
    plt.legend(loc='lower right')
    plt.xlabel('cycle')
    plt.yscale("log")
    plt.ylabel("Time [s]")
    # output_path = f'{dataset_name}_time.png'
    output_path = f'/mnt/stud/home/jcheng/scikit-activeml/tutorials/result/{dataset_name}_time.pdf'
    plt.savefig(output_path)


if __name__ == '__main__':
    parser = parse_argument()
    args = parser.parse_args()
    dataset_name = args.dataset
    n_cycles = args.n_cycles

    evaluate_experiment_csv_score(dataset_name)
