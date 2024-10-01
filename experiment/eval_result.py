import numpy as np
import pandas as pd
import csv

OUTPUT_PATH = '/Users/chengjiaying/PycharmProjects/scikit-activeml/experiment/output_image/'


batch_size_dict = {
        'letter': 156,
        'dopanim': 90,
        'agnews': 24,
    }


def get_mean_std(mean_list, std_list):
    mean_a = mean_list.mean()
    var_annotator_query = std_list * std_list
    var_a = var_annotator_query.mean()
    std_a = np.sqrt(var_a)
    return mean_a, std_a


def eval_result(
        datasets,
        instance_query_strategies,
        annotator_query_strategies,
        learning_strategies,
        n_annotator_list,
        batch_size_list,
        metric,
):
    results = [['S', 'T', 'R', 'W', 'dopanim', 'letter', 'agnews']]
    for b in batch_size_list:
        for idx_i, instance_query_strategy in enumerate(instance_query_strategies):
            for idx_a, annotator_query_strategy in enumerate(annotator_query_strategies):
                for idx_l, learning_strategy in enumerate(learning_strategies):
                    for idx_n, n_annotator_per_instance in enumerate(n_annotator_list):
                        if (annotator_query_strategy in ['trace-reg', 'geo-reg-f', 'geo-reg-w'] and
                                learning_strategy != annotator_query_strategy):
                            continue

                        means = []
                        stds = []
                        for dataset in datasets:
                            batch_size = batch_size_dict[dataset] * b

                            label = (f'{instance_query_strategy} '
                                     f'+ {annotator_query_strategy} '
                                     f'+ {learning_strategy} '
                                     f'+ {n_annotator_per_instance} '
                                     f'+ {batch_size}')
                            df = pd.read_csv(f'{OUTPUT_PATH}/result_{dataset}/{label}.csv')
                            metric_mean = df[f'{metric}_mean'].to_numpy()
                            metric_std = df[f'{metric}_std'].to_numpy()
                            mean_a, std_a = get_mean_std(metric_mean, metric_std)
                            mean_a = np.round(mean_a * 100, 2)
                            std_a = np.round(std_a * 100, 1)
                            means.append(mean_a)
                            stds.append(std_a)
                        if idx_n == 0:
                            results.append(['\hline'])
                            results.append(['\multirow{3}{*}{'+ instance_query_strategy + '} ', '\multirow{3}{*}{'+ annotator_query_strategy + '} ', '\multirow{3}{*}{'+ learning_strategy + '} ', n_annotator_per_instance, f'{means[0]} $\pm$ {stds[0]}', f'{means[1]} $\pm$ {stds[1]}', f'{means[2]} $\pm$ {stds[2]} end'])
                        else:
                            results.append(['x', 'x', 'x', n_annotator_per_instance, f'{means[0]} $\pm$ {stds[0]}', f'{means[1]} $\pm$ {stds[1]}', f'{means[2]} $\pm$ {stds[2]} end'])

    with open(f'{OUTPUT_PATH}/result_3.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(results)


if __name__ == '__main__':
    instance_query_strategies = ['random', 'gsx', 'uncertainty', 'coreset', 'clue', 'typiclust']
    annotator_query_strategies = ['random', 'round-robin', 'trace-reg', 'geo-reg-f', 'geo-reg-w']
    learning_strategies = ['majority-vote', 'trace-reg', 'geo-reg-f', 'geo-reg-w']

    dataset = 'dopanim'
    metric = 'misclassification'

    batch_size = batch_size_dict[dataset]

    eval_result(
        datasets=['dopanim', 'letter', 'agnews'],
        instance_query_strategies=instance_query_strategies,
        annotator_query_strategies=annotator_query_strategies,
        learning_strategies=learning_strategies,
        n_annotator_list=[1, 2, 3],
        batch_size_list=[1, 2],
        metric=metric,
    )

