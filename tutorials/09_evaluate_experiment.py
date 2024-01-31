import os
import mlflow
import matplotlib as mlp
import matplotlib.pyplot as plt
import warnings

import pandas as pd
mlp.rcParams["figure.facecolor"] = "white"
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    mlflow.set_tracking_uri(uri="/Users/chengjiaying/scikit-activeml/tutorials/tracking")
    # mlflow.set_tracking_uri(uri="file:///mnt/stud/home/jcheng/scikit-activeml/tutorials/tracking")

    experiment = mlflow.get_experiment_by_name("Evaluation-Active Learning")
    df = mlflow.search_runs(experiment_ids=experiment.experiment_id, output_format="pandas")

    df = df[['params.dataset', 'params.qs', 'params.batch_size', 'params.n_cycles', 'params.seed', 'artifact_uri']]
    print(len(df['params.seed'].unique()))
    for idx, row in df.iterrows():
        artifact = os.path.join(row.artifact_uri, 'result.csv')
        if os.path.exists(artifact):
            dataframe = pd.read_csv(artifact, index_col=0, on_bad_lines='skip')
            dataframe = dataframe.dropna()


