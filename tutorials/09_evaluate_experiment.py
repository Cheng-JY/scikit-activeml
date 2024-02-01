import os
import mlflow
import matplotlib as mlp
import matplotlib.pyplot as plt
import warnings
import numpy as np

import pandas as pd
mlp.rcParams["figure.facecolor"] = "white"
warnings.filterwarnings("ignore")
import argparse

def parse_argument():
    parser = argparse.ArgumentParser(description='Evaluate model performance')
    parser.add_argument('dataset', type=str, help='name of dataset')
    parser.add_argument('graph_type', type=str, help='accuracy or time')
    return parser

if __name__ == "__main__":
    parser = parse_argument()
    args = parser.parse_args()
    dataset_name = args.dataset
    graph_type = args.graph_type

    # mlflow.set_tracking_uri(uri="/Users/chengjiaying/scikit-activeml/tutorials/tracking")
    mlflow.set_tracking_uri(uri="file:///mnt/stud/home/jcheng/scikit-activeml/tutorials/tracking")

    experiment = mlflow.get_experiment_by_name("Evaluation-Active-Learning-Params")
    df = mlflow.search_runs(experiment_ids=experiment.experiment_id, output_format="pandas")

    df = df[['params.dataset', 'params.qs', 'params.batch_size', 'params.n_cycles', 'params.seed', 'artifact_uri']]

    df = df.loc[df['params.dataset'] == dataset_name]
    query_stragies = df['params.qs'].unique()

    for qs_name in query_stragies:
        print(qs_name)
        df_qs = df.loc[df['params.qs'] == qs_name]
        r = []
        for idx, row in df_qs.iterrows():
            artifact = os.path.join(row.artifact_uri, 'result.csv')
            artifact = artifact.split("file://")[1]
            if os.path.exists(artifact):
                result_qs = pd.read_csv(artifact, index_col=0)
                r.append(result_qs)
        results = pd.concat(r)
        result = results.groupby(['step'])[graph_type].agg(['mean', 'std']).set_axis(['mean', 'std'], axis=1)
        result_mean = result['mean'].to_numpy()
        result_std = result['std'].to_numpy()
        plt.errorbar(np.arange(len(result_mean)), result_mean, result_std,
                    label=f"({np.mean(result_mean):.4f}) {qs_name}", alpha=0.3)

    plt.title(dataset_name.upper())
    plt.legend(loc='lower right')
    plt.xlabel('cycle')
    plt.ylabel(graph_type)
    # output_path = f'{dataset_name}_{graph_type}.pdf'
    output_path = f'/mnt/stud/home/jcheng/scikit-activeml/tutorials/result_param/{dataset_name}_{graph_type}.pdf'
    plt.savefig(output_path)



