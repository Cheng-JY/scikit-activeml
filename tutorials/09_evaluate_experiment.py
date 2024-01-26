import mlflow
import matplotlib as mlp
import matplotlib.pyplot as plt
import warnings

import pandas as pd
mlp.rcParams["figure.facecolor"] = "white"
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
    experiment = mlflow.get_experiment_by_name("Evaluation-Active Learning")
    df = mlflow.search_runs(experiment_ids=experiment.experiment_id, output_format="pandas")

    df = df[["tags.dataset", "tags.model", "tags.qs", "tags.batch_size", "tags.n_cycles", "tags.seed", "tags.cycle", "metrics.score", "metrics.time"]]


