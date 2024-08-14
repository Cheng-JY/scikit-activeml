import os
import sys
import hydra
import mlflow
import matplotlib as mlp
import matplotlib.pyplot as plt
import warnings
import numpy as np

import pandas as pd

sys.path.append("/mnt/stud/home/jcheng/scikit-activeml/")
warnings.filterwarnings("ignore")


@hydra.main(config_path="config", config_name="config", version_base="1.1")
def main(cfg):

    running_device = 'server'

    if running_device == 'server':
        dataset_name = cfg['dataset']
    else:
        dataset_name = 'letter_perf'

    ml_flow_tracking = cfg['ml_flow_tracking']
    mlflow.set_tracking_uri(uri=ml_flow_tracking[f"tracking_file_path_{running_device}"])
    exp = mlflow.get_experiment_by_name(name=ml_flow_tracking["experiment_name"])

    df = mlflow.search_runs(experiment_ids=exp.experiment_id, output_format="pandas")

    df = df[['params.dataset_name', 'params.instance_query_strategy', 'params.annotator_query_strategy',
             'params.learning_strategy', 'params.batch_size', 'params.n_annotators_per_sample',
             'params.n_cycles', 'params.seed', 'artifact_uri']]

    df = df.loc[df['params.dataset_name'] == dataset_name]

    instance_query_strategies = df['params.instance_query_strategy'].unique()
    annotator_query_strategies = df['params.annotator_query_strategy'].unique()
    learning_strategies = df['params.learning_strategy'].unique()
    n_annotators_per_sample_list = df['params.n_annotators_per_sample'].unique()
    batch_size_list = df['params.batch_size'].unique()

    for iqs_name in instance_query_strategies:
        for aqs_name in annotator_query_strategies:
            for ls_name in learning_strategies:
                for n_per_sample in n_annotators_per_sample_list:
                    for batch_size in batch_size_list:

                        df_qs = df.loc[(df['params.instance_query_strategy'] == iqs_name) &
                                       (df['params.annotator_query_strategy'] == aqs_name) &
                                       (df['params.learning_strategy'] == ls_name) &
                                       (df['params.n_annotators_per_sample'] == n_per_sample) &
                                       (df['params.batch_size'] == batch_size)]
                        r = []
                        for idx, row in df_qs.iterrows():
                            artifact = os.path.join(row.artifact_uri, 'result.csv')
                            # artifact = artifact.split("file://")[1]
                            print(artifact)
                            print(os.path.exists(artifact))
                            if os.path.exists(artifact):
                                result_qs = pd.read_csv(artifact, index_col=None)
                                r.append(result_qs)

                        if len(r) != 0:
                            results = pd.concat(r)
                            mean_and_std = []
                            for column in results.columns:
                                result = results.groupby(['step'])[column].agg(['mean', 'std']).set_axis([f'{column}_mean', f'{column}_std'], axis=1)
                                mean_and_std.append(result)
                            r_mean_and_std = pd.concat(mean_and_std, axis=1)
                            label = (f'{iqs_name} '
                                     f'+ {aqs_name} '
                                     f'+ {ls_name} '
                                     f'+ {n_per_sample} '
                                     f'+ {batch_size}')
                            output_path = f'{cfg["output_file_path"][running_device]}/result_{dataset}/{label}.csv'
                            r_mean_and_std.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
