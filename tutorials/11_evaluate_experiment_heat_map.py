import os
import mlflow
import matplotlib as mlp
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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
    batch_size = 16

    # mlflow.set_tracking_uri(uri="/Users/chengjiaying/scikit-activeml/tutorials/tracking")
    mlflow.set_tracking_uri(uri="file:///mnt/stud/home/jcheng/scikit-activeml/tutorials/mlflow_tracking")

    experiment = mlflow.get_experiment_by_name("Evaluation-Active-Learning-Params")
    df = mlflow.search_runs(experiment_ids=experiment.experiment_id, output_format="pandas")

    df = df[['params.dataset', 'params.qs', 'params.batch_size', 'params.n_cycles', 'params.seed', 'artifact_uri']]

    df = df.loc[df['params.dataset'] == dataset_name]
    query_stragies = df['params.qs'].unique()
    colors = ["b", "g", "r", "c", "m", "k"]
    result_dict = {}

    for idx, qs_name in enumerate(query_stragies):
        print(qs_name)
        print(idx)
        print(colors[idx])
        color = colors[idx]
        df_qs = df.loc[df['params.qs'] == qs_name]
        r = []
        for idx, row in df_qs.iterrows():
            artifact = os.path.join(row.artifact_uri, 'result.csv')
            artifact = artifact.split("file://")[1]
            print(artifact)
            print(os.path.exists(artifact))
            if os.path.exists(artifact):
                result_qs = pd.read_csv(artifact, index_col=0)
                r.append(result_qs)
        results = pd.concat(r)
        result = results.groupby(['step'])[graph_type].agg(['mean', 'std']).set_axis(['mean', 'std'], axis=1)
        result_mean = result['mean'].to_numpy()
        result_dict[idx] = result_mean

    heat_map_numpy = np.zeros(shape=(6,6))

    for i in range(6):
        for j in range(6):
            if i == j:
                continue
            win_counter = 0
            i_algo_result = result_dict[i]
            j_algo_result = result_dict[j]
            for l in range(len(i_algo_result)):
                if i_algo_result[l] > j_algo_result[l]:
                    win_counter += 1
            heat_map_numpy[i, j] = win_counter / (len(i_algo_result))

    heat_map_sum = np.sum(heat_map_numpy, axis=1).reshape(-1, 1)

    fig, axs = plt.subplots(1, 2, sharey=True, gridspec_kw={'width_ratios': [6,1]})
    a1 = axs[0].imshow(heat_map_numpy, cmap="YlGn",
                       vmin=0.0, vmax=1.0, aspect='auto',
                       interpolation='nearest')
    a2 = axs[1].imshow(heat_map_sum, cmap="YlGn",
                       vmin=0.0, vmax=1.0, aspect='auto',
                       interpolation='nearest')

    axs.set_xticks(np.arange(len(query_stragies)), labels=query_stragies)
    axs.set_yticks(np.arange(len(query_stragies)), labels=query_stragies)

    plt.colorbar(a1)

    output_path = f'/mnt/stud/home/jcheng/scikit-activeml/tutorials/result_param/heatmap.pdf'
    plt.savefig(output_path)



