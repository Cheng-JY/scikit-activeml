import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ipywidgets as widgets
from IPython.core.display_functions import display

linestyles = ['solid', 'dotted', 'dashed', 'dashdot']
colors = ['red', 'blue', 'green', 'orange']


def plot_graph(
        dataset,
        question,
        instance_query_strategies,
        annotator_query_strategies,
        learning_strategies,
        n_annotator_list,
        batch_size,
        metric,
        name='',
):
    output_path = '/Users/chengjiaying/PycharmProjects/scikit-activeml/experiment/output_image/'

    for idx_i, instance_query_strategy in enumerate(instance_query_strategies):
        for idx_a, annotator_query_strategy in enumerate(annotator_query_strategies):
            for idx_l, learning_strategy in enumerate(learning_strategies):
                for idx_n, n_annotator_per_instance in enumerate(n_annotator_list):
                    if (annotator_query_strategy in ['trace-reg', 'geo-reg-f', 'geo-reg-w'] and
                            learning_strategy != annotator_query_strategy):
                        continue
                    # if (annotator_query_strategy in ['random', 'round-robin'] and
                    #         learning_strategy != 'majority-vote'):
                    #     continue
                    label = (f'{instance_query_strategy} '
                             f'+ {annotator_query_strategy} '
                             f'+ {learning_strategy} '
                             f'+ {n_annotator_per_instance} '
                             f'+ {batch_size}')
                    df = pd.read_csv(f'{output_path}/result_{dataset}/{label}.csv')
                    metric_mean = df[f'{metric}_mean'].to_numpy()
                    metric_std = df[f'{metric}_std'].to_numpy()
                    if question == 'RQ1':
                        plt.errorbar(np.arange(batch_size, (len(metric_mean) + 1) * batch_size, batch_size),
                                     metric_mean,
                                     metric_std,
                                     label=f"({np.mean(metric_mean):.4f}) {label}", alpha=0.3)
                    elif question == 'RQ2':
                        plt.errorbar(np.arange(batch_size, (len(metric_mean) + 1) * batch_size, batch_size),
                                     metric_mean,
                                     metric_std, color=colors[idx_a],
                                     label=f"({np.mean(metric_mean):.4f}) {label}", alpha=0.3)
                    elif question == 'RQ3':
                        plt.errorbar(np.arange(batch_size, (len(metric_mean) + 1) * batch_size, batch_size), metric_mean,
                                     metric_std, linestyle=linestyles[idx_l], color=colors[idx_a],
                                     label=f"({np.mean(metric_mean):.4f}) {label}", alpha=0.3)
                    elif question == 'RQ4':
                        plt.errorbar(np.arange(batch_size, (len(metric_mean) + 1) * batch_size, batch_size), metric_mean,
                                     metric_std, linestyle=linestyles[idx_n], color=colors[idx_a],
                                     label=f"({np.mean(metric_mean):.4f}) {label}", alpha=0.3)

    plt.legend(bbox_to_anchor=(1.05, 1), fontsize=6, loc='upper right', ncol=2)
    plt.tight_layout()
    plt.xlabel('# Labels queried')
    plt.ylabel(f"{metric}")
    title = f'{dataset}_{metric}_{question}_{batch_size}_{name}'
    plt.title(title)
    output_path = f'{output_path}/{dataset}_plot/{title}.pdf'
    plt.savefig(output_path, bbox_inches="tight")


def eval_RQ1(
        dataset,
        instance_query_strategies,
        annotator_query_strategies,
        learning_strategies,
        n_annotator_list,
        batch_size,
        metric,
):
    output_path = '/Users/chengjiaying/PycharmProjects/scikit-activeml/experiment/output_image/'

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
                    df = pd.read_csv(f'{output_path}/result_{dataset}/{label}.csv')
                    metric_mean = df[f'{metric}_mean'].to_numpy()
                    metric_std = df[f'{metric}_std'].to_numpy()
                    mean_instance_query.append(metric_mean)
                    std_instance_query.append(metric_std)
        mean_instance_query = np.asarray(mean_instance_query)
        std_instance_query = np.asarray(std_instance_query)

        mean_i = mean_instance_query.mean(axis=0)
        var_instance_query = std_instance_query * std_instance_query
        var_i = var_instance_query.mean(axis=0)
        std_i = np.sqrt(var_i)
        plt.errorbar(np.arange(batch_size, (len(mean_i) + 1) * batch_size, batch_size),
                     mean_i,
                     std_i,
                     label=f"({np.mean(mean_i):.4f}) {instance_query_strategy}", alpha=0.3)

    plt.legend(bbox_to_anchor=(1.05, 1), fontsize=6, loc='upper right', ncol=2)
    plt.tight_layout()
    plt.xlabel('# Labels queried')
    plt.ylabel(f"{metric}")
    title = f'{dataset}_{metric}_{question}'
    plt.title(title)
    output_path = f'{output_path}/{dataset}_plot/{title}.pdf'
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
    output_path = '/Users/chengjiaying/PycharmProjects/scikit-activeml/experiment/output_image/'

    for idx_a, annotator_query_strategy in enumerate(annotator_query_strategies):
        mean_annotator_query = []
        std_annotator_query = []
        for idx_i, instance_query_strategy in enumerate(instance_query_strategies):
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
                    df = pd.read_csv(f'{output_path}/result_{dataset}/{label}.csv')
                    metric_mean = df[f'{metric}_mean'].to_numpy()
                    metric_std = df[f'{metric}_std'].to_numpy()
                    mean_annotator_query.append(metric_mean)
                    std_annotator_query.append(metric_std)
        mean_annotator_query = np.asarray(mean_annotator_query)
        std_annotator_query = np.asarray(std_annotator_query)

        mean_a = mean_annotator_query.mean(axis=0)
        var_annotator_query = std_annotator_query * std_annotator_query
        var_a = var_annotator_query.mean(axis=0)
        std_a = np.sqrt(var_a)
        plt.errorbar(np.arange(batch_size, (len(mean_a) + 1) * batch_size, batch_size),
                     mean_a, std_a,
                     label=f"({np.mean(mean_a):.4f}) {annotator_query_strategy}", alpha=0.3)

    plt.legend(bbox_to_anchor=(1.05, 1), fontsize=6, loc='upper right', ncol=2)
    plt.tight_layout()
    plt.xlabel('# Labels queried')
    plt.ylabel(f"{metric}")
    title = f'{dataset}_{metric}_{question}'
    plt.title(title)
    output_path = f'{output_path}/{dataset}_plot/{title}.pdf'
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
    output_path = '/Users/chengjiaying/PycharmProjects/scikit-activeml/experiment/output_image/'

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
                    df = pd.read_csv(f'{output_path}/result_{dataset}/{label}.csv')
                    metric_mean = df[f'{metric}_mean'].to_numpy()
                    metric_std = df[f'{metric}_std'].to_numpy()
                    mean_learning_strategy.append(metric_mean)
                    std_learning_strategy.append(metric_std)
        mean_learning_strategy = np.asarray(mean_learning_strategy)
        std_learning_strategy = np.asarray(std_learning_strategy)

        mean_a = mean_learning_strategy.mean(axis=0)
        var_annotator_query = std_learning_strategy * std_learning_strategy
        var_a = var_annotator_query.mean(axis=0)
        std_a = np.sqrt(var_a)
        plt.errorbar(np.arange(batch_size, (len(mean_a) + 1) * batch_size, batch_size),
                     mean_a, std_a,
                     label=f"({np.mean(mean_a):.4f}) {learning_strategy}", alpha=0.3)

    plt.legend(bbox_to_anchor=(1.05, 1), fontsize=6, loc='upper right', ncol=2)
    plt.tight_layout()
    plt.xlabel('# Labels queried')
    plt.ylabel(f"{metric}")
    title = f'{dataset}_{metric}_{question}'
    plt.title(title)
    output_path = f'{output_path}/{dataset}_plot/{title}.pdf'
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
    output_path = '/Users/chengjiaying/PycharmProjects/scikit-activeml/experiment/output_image/'

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
                    df = pd.read_csv(f'{output_path}/result_{dataset}/{label}.csv')
                    metric_mean = df[f'{metric}_mean'].to_numpy()
                    metric_std = df[f'{metric}_std'].to_numpy()
                    mean_n_annotator.append(metric_mean)
                    std_n_annotator.append(metric_std)
        mean_n_annotator = np.asarray(mean_n_annotator)
        std_n_annotator = np.asarray(std_n_annotator)

        mean_a = mean_n_annotator.mean(axis=0)
        var_annotator_query = std_n_annotator * std_n_annotator
        var_a = var_annotator_query.mean(axis=0)
        std_a = np.sqrt(var_a)
        plt.errorbar(np.arange(batch_size, (len(mean_a) + 1) * batch_size, batch_size),
                     mean_a, std_a,
                     label=f"({np.mean(mean_a):.4f}) {n_annotator_per_instance}", alpha=0.3)

    plt.legend(bbox_to_anchor=(1.05, 1), fontsize=6, loc='upper right', ncol=2)
    plt.tight_layout()
    plt.xlabel('# Labels queried')
    plt.ylabel(f"{metric}")
    title = f'{dataset}_{metric}_{question}'
    plt.title(title)
    output_path = f'{output_path}/{dataset}_plot/{title}.pdf'
    plt.savefig(output_path, bbox_inches="tight")


if __name__ == '__main__':
    metrics = ['misclassification', 'error_annotation_rate']
    for i in range(10):
        metrics.append(f"Number_of_annotations_{i}")
        metrics.append(f"Number_of_correct_annotation_{i}")

    instance_query_strategies = ['random', 'uncertainty', 'coreset']
    annotator_query_strategies = ['random', 'round-robin', 'trace-reg', 'geo-reg-f', 'geo-reg-w']
    learning_strategies = ['majority-vote', 'trace-reg', 'geo-reg-f', 'geo-reg-w']

    dataset = 'letter'
    question = 'RQ4'
    metric = 'misclassification'
    intelligent = False
    batch_size = 312
    # RQ1: Instance selecting 312, 156
    if question == 'RQ1':
        # plot_graph(
        #     dataset=dataset,
        #     question=question,
        #     instance_query_strategies=['random', 'gsx', 'uncertainty', 'coreset', 'clue', 'typiclust'],
        #     annotator_query_strategies=['round-robin'],
        #     learning_strategies=['trace-reg'],
        #     n_annotator_list=[1],
        #     batch_size=batch_size,
        #     metric=metric,
        # )

        eval_RQ1(
            dataset=dataset,
            instance_query_strategies=['random', 'gsx', 'uncertainty', 'coreset', 'clue', 'typiclust'],
            annotator_query_strategies=['random', 'round-robin', 'trace-reg', 'geo-reg-f', 'geo-reg-w'],
            learning_strategies=['majority-vote', 'trace-reg', 'geo-reg-f', 'geo-reg-w'],
            n_annotator_list=[1],
            batch_size=batch_size,
            metric=metric,
        )
    elif question == 'RQ2':
        # plot_graph(
        #     dataset=dataset,
        #     question=question,
        #     instance_query_strategies=['uncertainty'],
        #     annotator_query_strategies=['random', 'round-robin', 'geo-reg-f'],
        #     learning_strategies=['geo-reg-f'],
        #     n_annotator_list=[1],
        #     batch_size=batch_size,
        #     metric=metric,
        # )

        eval_RQ2(
            dataset=dataset,
            instance_query_strategies=['random', 'gsx', 'uncertainty', 'coreset', 'clue', 'typiclust'],
            annotator_query_strategies=['random', 'round-robin', 'trace-reg', 'geo-reg-f', 'geo-reg-w'],
            learning_strategies=['trace-reg', 'geo-reg-f', 'geo-reg-w'],
            n_annotator_list=[1, 2, 3],
            batch_size=batch_size,
            metric=metric,
        )

    elif question == 'RQ3':
        # plot_graph(
        #     dataset=dataset,
        #     question=question,
        #     instance_query_strategies=['random'],
        #     annotator_query_strategies=['random', 'round-robin'],
        #     learning_strategies=['majority-vote', 'trace-reg', 'geo-reg-f', 'geo-reg-w'],
        #     n_annotator_list=[1],
        #     batch_size=batch_size,
        #     metric=metric,
        # )

        eval_RQ3(
            dataset=dataset,
            instance_query_strategies=['random', 'gsx', 'uncertainty', 'coreset', 'clue', 'typiclust'],
            annotator_query_strategies=['random', 'round-robin'],
            learning_strategies=['majority-vote', 'trace-reg', 'geo-reg-f', 'geo-reg-w'],
            n_annotator_list=[1, 2, 3],
            batch_size=batch_size,
            metric=metric,
        )

    elif question == 'RQ4':
        # if not intelligent:
        #     plot_graph(
        #         dataset=dataset,
        #         question=question,
        #         instance_query_strategies=['random'],
        #         annotator_query_strategies=['random', 'round-robin'],
        #         learning_strategies=['majority-vote'],
        #         n_annotator_list=[1, 2, 3],
        #         batch_size=batch_size,
        #         metric=metric,
        #         name='not_intelligent',
        #     )
        # else:
        #     plot_graph(
        #         dataset=dataset,
        #         question=question,
        #         instance_query_strategies=['random'],
        #         annotator_query_strategies=['trace-reg', 'geo-reg-f', 'geo-reg-w'],
        #         learning_strategies=['trace-reg', 'geo-reg-f', 'geo-reg-w'],
        #         n_annotator_list=[1, 2, 3],
        #         batch_size=batch_size,
        #         metric=metric,
        #         name='intelligent',
        #     )

        eval_RQ4(
            dataset=dataset,
            instance_query_strategies=['random', 'gsx', 'uncertainty', 'coreset', 'clue', 'typiclust'],
            annotator_query_strategies=['random', 'round-robin', 'trace-reg', 'geo-reg-f', 'geo-reg-w'],
            learning_strategies=['majority-vote', 'trace-reg', 'geo-reg-f', 'geo-reg-w'],
            n_annotator_list=[1, 2, 3],
            batch_size=batch_size,
            metric=metric,
        )

