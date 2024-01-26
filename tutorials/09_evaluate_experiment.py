import mlflow
import matplotlib as mlp
import matplotlib.pyplot as plt
import warnings

import pandas as pd
mlp.rcParams["figure.facecolor"] = "white"
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    #mlflow.set_tracking_uri(uri="/Users/chengjiaying/scikit-activeml/tutorials/tracking")
    experiment = mlflow.get_experiment_by_name("Evaluation-Active Learning")
    df = mlflow.search_runs(experiment_ids=experiment.experiment_id, output_format="pandas")

    for r_idx, r in df.iterrows():
        print(r["metrics.score"])
