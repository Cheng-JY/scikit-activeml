import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ipywidgets as widgets
from IPython.core.display_functions import display


def plot_graph(
        instance_query_strategies,
        annotator_query_strategies,
        learning_strategies,
        n_annotator_list,
        metric,
        name,
):
    output_path = '/Users/chengjiaying/PycharmProjects/scikit-activeml/experiment/output_image/'

    batch_size = 256

    for instance_query_strategy in instance_query_strategies:
        for annotator_query_strategy in annotator_query_strategies:
            for learning_strategy in learning_strategies:
                for n_annotator_per_instance in n_annotator_list:
                    if (annotator_query_strategy in ['trace-reg', 'geo-reg-f', 'geo-reg-w'] and
                            learning_strategy != annotator_query_strategy):
                        continue
                    # if (annotator_query_strategy in ['random', 'round-robin'] and
                    #         learning_strategy != 'majority-vote'):
                    #     continue
                    label = (f'{instance_query_strategy} '
                             f'+ {annotator_query_strategy} '
                             f'+ {learning_strategy} '
                             f'+ {n_annotator_per_instance}')
                    df = pd.read_csv(f'{output_path}/result_letter/{label}.csv')
                    metric_mean = df[f'{metric}_mean'].to_numpy()
                    metric_std = df[f'{metric}_std'].to_numpy()
                    plt.errorbar(np.arange(batch_size, (len(metric_mean) + 1) * batch_size, batch_size), metric_mean,
                                 metric_std,
                                 label=f"({np.mean(metric_mean):.4f}) {label}", alpha=0.3)

    plt.legend(bbox_to_anchor=(1.05, 1), fontsize=6, loc='upper right', ncol=2)
    plt.tight_layout()
    plt.xlabel('# Labels queried')
    plt.ylabel(f"{metric}")
    title = f'letter_{metric}_{name}'
    plt.title(title)
    output_path = f'{output_path}/letter_plot/{title}.pdf'
    plt.savefig(output_path, bbox_inches="tight")


if __name__ == '__main__':
    metrics = ['misclassification', 'error_annotation_rate']
    for i in range(10):
        metrics.append(f"Number_of_annotations_{i}")
        metrics.append(f"Number_of_correct_annotation_{i}")

    instance_query_strategies = ['random', 'uncertainty', 'coreset']
    annotator_query_strategies = ['random', 'round-robin', 'trace-reg', 'geo-reg-f', 'geo-reg-w']
    learning_strategies = ['majority-vote', 'trace-reg', 'geo-reg-f', 'geo-reg-w']

    plot_graph(
        instance_query_strategies=['random'],
        annotator_query_strategies=['trace-reg', 'geo-reg-f', 'geo-reg-w'],
        learning_strategies=learning_strategies,
        n_annotator_list=[1, 2, 3],
        metric='misclassification',
        name='RQ4-intelligent'
    )
