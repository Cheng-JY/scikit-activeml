import os
import sys
import warnings

sys.path.append("/mnt/stud/home/jcheng/scikit-activeml/")
warnings.filterwarnings("ignore")

import hydra
import mlflow
import matplotlib as mlp
import matplotlib.pyplot as plt
import warnings
import numpy as np

import pandas as pd
import argparse

mlp.rcParams["figure.facecolor"] = "white"
warnings.filterwarnings("ignore")


def parse_argument():
    parser = argparse.ArgumentParser(description='Evaluate model performance')
    parser.add_argument('dataset', type=str, help='name of dataset')
    parser.add_argument('instance_query_strategy', type=str, help='name of instance query strategy')
    parser.add_argument('annotator_query_strategy', type=str, help='name of annotator query strategy')
    parser.add_argument('batch_size', type=int, help='batch size')
    parser.add_argument('n_annotators_per_sample', type=int, help='n_annotators_per_sample')
    parser.add_argument('n_cycles', type=int, help='number of cycles')
    parser.add_argument('graph_type', type=str, help='graph_type')
    return parser


def get_parse_argument(parser):
    args = parser.parse_args()
    experiment_params = {
        'dataset_name': args.dataset,
        'instance_query_strategy': args.instance_query_strategy,
        'annotator_query_strategy': args.annotator_query_strategy,
        'batch_size': args.batch_size,
        'n_annotators_per_sample': args.n_annotators_per_sample,
        'n_cycles': args.n_cycles,
        'graph_type': args.graph_type,
    }
    return experiment_params


@hydra.main(config_path="config", config_name="config", version_base="1.1")
def main(cfg):
    parser = parse_argument()
    experiment_params = get_parse_argument(parser)

    # experiment_params = {
    #     'dataset_name': 'letter',
    #     'instance_query_strategy': 'random',
    #     'annotator_query_strategy': 'random',
    #     'batch_size': 256,
    #     'n_annotators_per_sample': 1,
    #     'n_cycles': 25,
    #     'graph_type': 'misclassification',
    # }

    ml_flow_tracking = cfg['ml_flow_tracking']
    mlflow.set_tracking_uri(uri=ml_flow_tracking['tracking_file_path_server'])
    exp = mlflow.get_experiment_by_name(name=ml_flow_tracking["experiment_name"])

    df = mlflow.search_runs(experiment_ids=exp.experiment_id, output_format="pandas")

    df = df[['params.dataset_name', 'params.instance_query_strategy', 'params.annotator_query_strategy',
             'params.batch_size', 'params.n_annotators_per_sample', 'params.n_cycles', 'params.seed',
             'artifact_uri']]

    df = df.loc[df['params.dataset_name'] == experiment_params['dataset_name']]

    df = df.loc[df['params.dataset_name'] == experiment_params["dataset_name"]]
    instance_query_strategies = df['params.instance_query_strategy'].unique()
    annotator_query_strategies = df['params.annotator_query_strategy'].unique()
    n_annotators_per_sample_list = df['params.n_annotators_per_sample'].unique()

    for iqs_name in instance_query_strategies:
        for aqs_name in annotator_query_strategies:
            for n_per_sample in n_annotators_per_sample_list:

                df_qs = df.loc[(df['params.instance_query_strategy'] == iqs_name) &
                               (df['params.annotator_query_strategy'] == aqs_name) &
                               (df['params.n_annotators_per_sample'] == n_per_sample)]
                r = []
                for idx, row in df_qs.iterrows():
                    artifact = os.path.join(row.artifact_uri, 'result.csv')
                    # artifact = artifact.split("file://")[1]
                    print(artifact)
                    print(os.path.exists(artifact))
                    if os.path.exists(artifact):
                        result_qs = pd.read_csv(artifact, index_col=0)
                        r.append(result_qs)
                results = pd.concat(r)
                result = results.groupby(['step'])[experiment_params['graph_type']].agg(['mean', 'std']).set_axis(['mean', 'std'], axis=1)
                result_mean = result['mean'].to_numpy()
                result_std = result['std'].to_numpy()
                label = (f'{experiment_params["instance_query_strategy"]} '
                         f'+ {experiment_params["annotator_query_strategy"]} '
                         f'+ {experiment_params["n_annotators_per_sample"]}')
                plt.errorbar(np.arange(16, (len(result_mean) + 1) * 16, 16), result_mean, result_std,
                             label=f"({np.mean(result_mean):.4f}) {label}", alpha=0.3)

    plt.legend(bbox_to_anchor=(0.5, -0.35), loc='lower center', ncol=2)
    plt.tight_layout()
    plt.xlabel('# Labels queried')
    plt.ylabel(f"{experiment_params['graph_type']}")
    title = f'{experiment_params["dataset_name"]}'
    plt.title(title)
    output_path = f'{cfg["output_file_path"]["server"]}/{title}.pdf'
    plt.savefig(output_path, bbox_inches="tight")


if __name__ == "__main__":
    main()
