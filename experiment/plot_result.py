import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plot_utils import *

linestyles = ['solid', 'dotted', 'dashed', 'dashdot']
colors = ['red', 'blue', 'green', 'orange']
n_annotators_dict = {
    'letter': 10,
    'dopanim': 20,
    'agnews': 20,
}

OUTPUT_PATH = '/Users/chengjiaying/PycharmProjects/scikit-activeml/experiment/output_image/'

idx_linestyle_dict = {
    'random': 0,
    'round-robin': 1,
    'trace-reg': 2,
    'geo-reg-f': 2,
    'geo-reg-w': 2,
}


def get_mean_std(mean_list, std_list):
    mean_a = mean_list.mean(axis=0)
    var_annotator_query = std_list * std_list
    var_a = var_annotator_query.mean(axis=0)
    std_a = np.sqrt(var_a)
    return mean_a, std_a


def get_init_batch_size(dataset, batch_size):
    init_batch_size = n_annotators_dict[dataset] * batch_size
    return init_batch_size


def eval_RQ1(
        dataset,
        instance_query_strategies,
        annotator_query_strategies,
        learning_strategies,
        n_annotator_list,
        batch_size,
        metric,
):
    for idx_i, instance_query_strategy in enumerate(instance_query_strategies):
        mean_instance_query = []
        std_instance_query = []
        for idx_a, annotator_query_strategy in enumerate(annotator_query_strategies):
            for idx_l, learning_strategy in enumerate(learning_strategies):
                for idx_n, n_annotator_per_instance in enumerate(n_annotator_list):
                    if (annotator_query_strategy in ['trace-reg', 'geo-reg-f', 'geo-reg-w'] and
                            learning_strategy != annotator_query_strategy):
                        continue
                    label = (f'{instance_query_strategy} '
                             f'+ {annotator_query_strategy} '
                             f'+ {learning_strategy} '
                             f'+ {n_annotator_per_instance} '
                             f'+ {batch_size}')
                    df = pd.read_csv(f'{OUTPUT_PATH}/result_{dataset}/{label}.csv')
                    metric_mean = df[f'{metric}_mean'].to_numpy()
                    metric_std = df[f'{metric}_std'].to_numpy()
                    mean_instance_query.append(metric_mean)
                    std_instance_query.append(metric_std)
        mean_instance_query = np.asarray(mean_instance_query)
        std_instance_query = np.asarray(std_instance_query)

        mean_i, std_i = get_mean_std(mean_instance_query, std_instance_query)
        init_batch_size = get_init_batch_size(dataset, batch_size)
        plt.plot(np.arange(init_batch_size, len(mean_i) * batch_size + init_batch_size, batch_size),
                 mean_i, label=f"({np.mean(mean_i):.4f}) {instance_query_strategy}", alpha=0.3)

    plt.legend(bbox_to_anchor=(0.5, -0.35), fontsize=12, loc='lower center', ncol=3)
    plt.tight_layout()
    plt.xlabel('# Annotations queried')
    plt.ylabel(f"{metric}")
    title = f'{dataset}_{metric}_RQ1_{batch_size}'
    output_path = f'{OUTPUT_PATH}/{dataset}_plot/{title}.pdf'
    plt.savefig(output_path, bbox_inches="tight")


def eval_RQ2(
        dataset,
        instance_query_strategies,
        annotator_query_strategies,
        learning_strategies,
        n_annotator_list,
        batch_size,
        metric,
):
    for idx_l, learning_strategy in enumerate(learning_strategies):
        mean_learning_strategy = []
        std_learning_strategy = []
        for idx_i, instance_query_strategy in enumerate(instance_query_strategies):
            for idx_a, annotator_query_strategy in enumerate(annotator_query_strategies):
                for idx_n, n_annotator_per_instance in enumerate(n_annotator_list):
                    if (annotator_query_strategy in ['trace-reg', 'geo-reg-f', 'geo-reg-w'] and
                            learning_strategy != annotator_query_strategy):
                        continue
                    label = (f'{instance_query_strategy} '
                             f'+ {annotator_query_strategy} '
                             f'+ {learning_strategy} '
                             f'+ {n_annotator_per_instance} '
                             f'+ {batch_size}')
                    df = pd.read_csv(f'{OUTPUT_PATH}/result_{dataset}/{label}.csv')
                    metric_mean = df[f'{metric}_mean'].to_numpy()
                    metric_std = df[f'{metric}_std'].to_numpy()
                    mean_learning_strategy.append(metric_mean)
                    std_learning_strategy.append(metric_std)
        mean_learning_strategy = np.asarray(mean_learning_strategy)
        std_learning_strategy = np.asarray(std_learning_strategy)

        mean_a, std_a = get_mean_std(mean_learning_strategy, std_learning_strategy)
        init_batch_size = get_init_batch_size(dataset, batch_size)
        plt.plot(np.arange(init_batch_size, len(mean_a) * batch_size + init_batch_size, batch_size),
                     mean_a, label=f"({np.mean(mean_a):.4f}) {learning_strategy}", alpha=0.3)

    plt.legend(bbox_to_anchor=(0.5, -0.35), fontsize=12, loc='lower center', ncol=2)
    plt.tight_layout()
    plt.xlabel('# Annotation queried')
    plt.ylabel(f"{metric}")
    title = f'{dataset}_{metric}_RQ2_{batch_size}'
    output_path = f'{OUTPUT_PATH}/{dataset}_plot/{title}.pdf'
    plt.savefig(output_path, bbox_inches="tight")


def eval_RQ3(
        dataset,
        instance_query_strategies,
        annotator_query_strategies,
        learning_strategies,
        n_annotator_list,
        batch_size,
        metric,
):
    for idx_l, learning_strategy in enumerate(learning_strategies):
        for idx_a, annotator_query_strategy in enumerate(annotator_query_strategies):
            for idx_i, instance_query_strategy in enumerate(instance_query_strategies):
                for idx_n, n_annotator_per_instance in enumerate(n_annotator_list):
                    if (annotator_query_strategy in ['trace-reg', 'geo-reg-f', 'geo-reg-w'] and
                            learning_strategy != annotator_query_strategy):
                        continue
                    metric_mean, metric_std, label = get_metric(dataset, instance_query_strategy,
                                                                annotator_query_strategy,
                                                                learning_strategy, n_annotator_per_instance, batch_size,
                                                                metric)

                    idx_linestyle = idx_linestyle_dict[annotator_query_strategy]

                    plt.plot(np.arange(batch_size, (len(metric_mean) + 1) * batch_size, batch_size),
                             metric_mean, color=colors[idx_l], linestyle=linestyles[idx_linestyle],
                             label=f"({np.mean(metric_mean):.4f}) {annotator_query_strategy} - {learning_strategy}",
                             alpha=0.3)

    plt.legend(bbox_to_anchor=(0.5, -0.35), fontsize=12, loc='lower center', ncol=3)
    plt.tight_layout()
    plt.xlabel('# Annotation queried')
    plt.ylabel(f"{metric}")
    title = f'{dataset}_{metric}_RQ3_{batch_size}'
    output_path = f'{OUTPUT_PATH}/{dataset}_plot/{title}.pdf'
    plt.savefig(output_path, bbox_inches="tight")


def eval_RQ4(
        dataset,
        instance_query_strategies,
        annotator_query_strategies,
        learning_strategies,
        n_annotator_list,
        batch_size,
        metric,
):
    for idx_n, n_annotator_per_instance in enumerate(n_annotator_list):
        mean_n_annotator = []
        std_n_annotator = []
        for idx_i, instance_query_strategy in enumerate(instance_query_strategies):
            for idx_a, annotator_query_strategy in enumerate(annotator_query_strategies):
                for idx_l, learning_strategy in enumerate(learning_strategies):
                    if (annotator_query_strategy in ['trace-reg', 'geo-reg-f', 'geo-reg-w'] and
                            learning_strategy != annotator_query_strategy):
                        continue
                    label = (f'{instance_query_strategy} '
                             f'+ {annotator_query_strategy} '
                             f'+ {learning_strategy} '
                             f'+ {n_annotator_per_instance} '
                             f'+ {batch_size}')
                    df = pd.read_csv(f'{OUTPUT_PATH}/result_{dataset}/{label}.csv')
                    metric_mean = df[f'{metric}_mean'].to_numpy()
                    metric_std = df[f'{metric}_std'].to_numpy()
                    mean_n_annotator.append(metric_mean)
                    std_n_annotator.append(metric_std)
        mean_n_annotator = np.asarray(mean_n_annotator)
        std_n_annotator = np.asarray(std_n_annotator)

        mean_a, std_a = get_mean_std(mean_n_annotator, std_n_annotator)
        plt.errorbar(np.arange(batch_size, (len(mean_a) + 1) * batch_size, batch_size),
                     mean_a, std_a / np.sqrt(5),
                     label=f"({np.mean(mean_a):.4f}) {n_annotator_per_instance}", alpha=0.3)

    plt.legend(bbox_to_anchor=(0.5, -0.3), fontsize=12, loc='lower center', ncol=3)
    plt.tight_layout()
    plt.xlabel('# Annotations queried')
    plt.ylabel(f"{metric}")
    title = f'{dataset}_{metric}_RQ4_{batch_size}'
    output_path = f'{OUTPUT_PATH}/{dataset}_plot/{title}.pdf'
    plt.savefig(output_path, bbox_inches="tight")


if __name__ == '__main__':
    metrics = ['misclassification', 'error_annotation_rate']
    for i in range(10):
        metrics.append(f"Number_of_annotations_{i}")
        metrics.append(f"Number_of_correct_annotation_{i}")

    instance_query_strategies = ['random', 'uncertainty', 'coreset']
    annotator_query_strategies = ['random', 'round-robin', 'trace-reg', 'geo-reg-f', 'geo-reg-w']
    learning_strategies = ['majority-vote', 'trace-reg', 'geo-reg-f', 'geo-reg-w']

    dataset = 'agnews'
    question = 'RQ1'
    metric = 'misclassification'
    batch_size_dict = {
        'letter': 156,
        'dopanim': 90,
    }
    batch_size = batch_size_dict[dataset] * 2

    if question == 'RQ1':
        eval_RQ1(
            dataset=dataset,
            instance_query_strategies=['random', 'gsx', 'uncertainty', 'coreset', 'clue', 'typiclust'],
            annotator_query_strategies=['trace-reg'],
            learning_strategies=['trace-reg'],
            n_annotator_list=[1],
            batch_size=batch_size,
            metric=metric,
        )
    elif question == 'RQ2':
        eval_RQ2(
            dataset=dataset,
            instance_query_strategies=['typiclust'],
            annotator_query_strategies=['round-robin'],
            learning_strategies=['majority-vote', 'trace-reg', 'geo-reg-f', 'geo-reg-w'],
            n_annotator_list=[1],
            batch_size=batch_size,
            metric=metric,
        )
    elif question == 'RQ3':
        eval_RQ3(
            dataset=dataset,
            instance_query_strategies=['uncertainty'],
            annotator_query_strategies=['random', 'round-robin', 'trace-reg', 'geo-reg-f', 'geo-reg-w'],
            learning_strategies=['trace-reg', 'geo-reg-f', 'geo-reg-w'],
            n_annotator_list=[1],
            batch_size=batch_size,
            metric=metric,
        )
    elif question == 'RQ4':
        eval_RQ4(
            dataset=dataset,
            instance_query_strategies=['uncertainty'],
            annotator_query_strategies=['trace-reg'],
            learning_strategies=['trace-reg'],
            n_annotator_list=[1, 2, 3],
            batch_size=batch_size,
            metric=metric,
        )
