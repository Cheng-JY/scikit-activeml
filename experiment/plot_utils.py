import pandas as pd


def get_metric(
        dataset,
        instance_query_strategy,
        annotator_query_strategy,
        learning_strategy,
        n_annotator_per_instance,
        batch_size,
        metric,
):
    output_path = '/Users/chengjiaying/PycharmProjects/scikit-activeml/experiment/output_image/'

    label = (f'{instance_query_strategy} '
             f'+ {annotator_query_strategy} '
             f'+ {learning_strategy} '
             f'+ {n_annotator_per_instance} '
             f'+ {batch_size}')

    df = pd.read_csv(f'{output_path}/result_{dataset}/{label}.csv')
    metric_mean = df[f'{metric}_mean'].to_numpy()
    metric_std = df[f'{metric}_std'].to_numpy()
    return metric_mean, metric_std, label
