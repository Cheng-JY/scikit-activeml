import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.colors as mcolors
import numpy as np
from plot_utils import *

OUTPUT_PATH = '/Users/chengjiaying/PycharmProjects/scikit-activeml/experiment/output_image'


def get_color(value):
    if value > 0.5:
        return "white"
    else:
        return "black"


def plot_heatmap(
        dataset,
        heat_map_numpy,
        heat_map_sum,
        strategies,
        batch_size,
        metric,
        question
):
    fig, axs = plt.subplots(1, 2, sharey=True, gridspec_kw={'width_ratios': [6, 1]})
    a1 = axs[0].imshow(heat_map_numpy, cmap="YlGnBu",
                       vmin=0.0, vmax=1.0, aspect='auto',
                       interpolation='nearest')
    a2 = axs[1].imshow(heat_map_sum, cmap="YlGnBu",
                       vmin=0.0, vmax=1.0, aspect='auto',
                       interpolation='nearest')

    axs[0].set_xticks(np.arange(len(strategies)), labels=strategies)
    plt.setp(axs[0].get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    axs[0].set_yticks(np.arange(len(strategies)), labels=strategies)

    axs[1].set_yticks(np.arange(len(strategies)), labels=strategies)
    axs[1].set_xticks(np.arange(1), ['average'])

    plt.colorbar(a2)

    for i in range(6):
        for j in range(6):
            text = axs[0].text(j, i, f"({heat_map_numpy[i, j]:.2f})",
                               ha="center", va="center", color=get_color(heat_map_numpy[i, j]))

    for i in range(6):
        text1 = axs[1].text(0, i, f"({heat_map_sum[i, 0]:.2f})",
                            ha="center", va="center", color=get_color(heat_map_sum[i, 0]))

    plt.title(dataset, loc="center")
    path = f'{OUTPUT_PATH}/heatmap/{dataset}_{metric}_{question}_{batch_size}.pdf'
    plt.savefig(path, bbox_inches="tight")


def pairwise_comparison_RQ1(
        dataset,
        instance_query_strategies,
        annotator_query_strategies,
        learning_strategies,
        n_annotator_list,
        batch_size,
        metric,
):
    heat_map_numpy = np.zeros(shape=(len(instance_query_strategies), len(instance_query_strategies)))

    for idx_i, instance_query_strategy_i in enumerate(instance_query_strategies):
        for idx_j, instance_query_strategy_j in enumerate(instance_query_strategies):
            if idx_i == idx_j:
                continue
            win_counter = 0
            sum_counter = 0
            for idx_a, annotator_query_strategy in enumerate(annotator_query_strategies):
                for idx_l, learning_strategy in enumerate(learning_strategies):
                    for idx_n, n_annotator_per_instance in enumerate(n_annotator_list):
                        if (annotator_query_strategy in ['trace-reg', 'geo-reg-f', 'geo-reg-w'] and
                                learning_strategy != annotator_query_strategy):
                            continue
                        sum_counter += 1
                        metric_mean_i, metric_std_i, label_i = get_metric(dataset, instance_query_strategy_i,
                                                                          annotator_query_strategy, learning_strategy,
                                                                          n_annotator_per_instance, batch_size, metric)

                        metric_mean_j, metric_std_j, label_j = get_metric(dataset, instance_query_strategy_j,
                                                                          annotator_query_strategy, learning_strategy,
                                                                          n_annotator_per_instance, batch_size, metric)

                        for l in range(len(metric_mean_i)):
                            t_stat, p_value = stats.ttest_ind_from_stats(metric_mean_i[l], metric_std_i[l], 5,
                                                                         metric_mean_j[l], metric_std_j[l], 5)
                            if p_value < 0.05 and metric_std_i[l] > metric_std_j[l]:
                                win_counter += 1 / (len(metric_mean_i))

            heat_map_numpy[idx_i, idx_j] = win_counter / sum_counter

    heat_map_sum = np.sum(heat_map_numpy, axis=1).reshape(-1, 1) / (len(instance_query_strategies) - 1)

    return heat_map_numpy, heat_map_sum


if __name__ == '__main__':
    dataset = 'dopanim'
    instance_query_strategies = ['random', 'gsx', 'uncertainty', 'coreset', 'clue', 'typiclust']
    annotator_query_strategies = ['random', 'round-robin', 'trace-reg', 'geo-reg-f', 'geo-reg-w']
    learning_strategies = ['majority-vote', 'trace-reg', 'geo-reg-f', 'geo-reg-w']
    n_annotator_list = [1, 2, 3]
    batch_size_dict = {
        'letter': 156,
        'dopanim': 90,
    }
    batch_size = batch_size_dict[dataset] * 2
    metric = 'misclassification'
    question = 'RQ1'

    if question == 'RQ1':
        heat_map_numpy, heat_map_sum = pairwise_comparison_RQ1(
            dataset=dataset,
            instance_query_strategies=['random', 'gsx', 'uncertainty', 'coreset', 'clue', 'typiclust'],
            annotator_query_strategies=['random', 'round-robin'],
            learning_strategies=['majority-vote', 'trace-reg', 'geo-reg-f', 'geo-reg-w'],
            n_annotator_list=[1, 2, 3],
            batch_size=batch_size,
            metric=metric,
        )
        strategies = instance_query_strategies


    plot_heatmap(
        dataset=dataset,
        heat_map_numpy=heat_map_numpy,
        heat_map_sum=heat_map_sum,
        strategies=strategies,
        batch_size=batch_size,
        metric=metric,
        question=question
    )
