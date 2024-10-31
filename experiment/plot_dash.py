import plotly.express as px
import plotly.graph_objects as go

from dash import Dash, dcc, html, Input, Output
from plot_utils import *
from pairwise_comparison import *

app = Dash(__name__)

# --Import and clean data (importing csv into pandas)
# I will do this inside the plot function
n_classes_dict = {
    'letter': 26,
    'dopanim': 15,
    'agnews': 4,
}

OUTPUT_PATH = '/Users/chengjiaying/PycharmProjects/scikit-activeml/experiment/output_image'

# App layout
app.layout = html.Div([

    # Title
    html.H1("Results", style={'textAlign': 'center'}),

    # Dash Component
    html.Div([
        "Research Question",
        dcc.Dropdown(id='research_question',
                     options=['RQ1', 'RQ2', 'RQ3', 'RQ4'],
                     multi=False,
                     value='RQ1')
    ]),

    html.Div([
        "Dataset",
        dcc.Dropdown(id='dataset',
                     options=['dopanim', 'letter', 'agnews'],
                     multi=False,
                     value='dopanim')
    ]),

    html.Div([
        "Instance Selection Strategies",
        dcc.Dropdown(id='instance_selection',
                     options=['random', 'gsx', 'uncertainty', 'coreset', 'clue', 'typiclust'],
                     multi=True,
                     value=['random'],)
    ]),

    html.Div([
        "Annotator Assignment Strategies",
        dcc.Dropdown(id='annotator_assignment',
                     options=['random', 'round-robin', 'intelligent'],
                     multi=True,
                     value=['random'],)
    ]),

    html.Div([
        "Noisy Annotation Handling Techniques",
        dcc.Dropdown(id='learning_strategy',
                     options=['majority-vote', 'trace-reg', 'geo-reg-f', 'geo-reg-w'],
                     multi=True,
                     value=['majority-vote'],)
    ]),

    html.Div([
        "Number of annotation per instance",
        dcc.Dropdown(id='n_annotator',
                     options=[
                         {'label': '1', 'value': 1},
                         {'label': '2', 'value': 2},
                         {'label': '3', 'value': 3}],
                     multi=True,
                     value=[1],)
    ]),

    html.Div([
        "Number of instances per class",
        dcc.Dropdown(id='batch_size',
                     options=[
                         {'label': '6', 'value': 6},
                         {'label': '12', 'value': 12}],
                     multi=False,
                     value=12,)
    ]),

    html.Div([
        "Metric",
        dcc.Dropdown(id='metric',
                     options=['misclassification', 'error_annotation_rate'],
                     multi=False,
                     value='misclassification', )
    ]),

    html.Br(),

    dcc.Graph(id='learning_curve', figure={}),
])


# Connect the ploty graph with Dash Components
@app.callback(
    Output(component_id='learning_curve', component_property='figure'),
    [Input(component_id='research_question', component_property='value'),
     Input(component_id='dataset', component_property='value'),
     Input(component_id='instance_selection', component_property='value'),
     Input(component_id='annotator_assignment', component_property='value'),
     Input(component_id='learning_strategy', component_property='value'),
     Input(component_id='n_annotator', component_property='value'),
     Input(component_id='batch_size', component_property='value'),
     Input(component_id='metric', component_property='value')]
)
def update_learning_curve(rq, dataset, instance_selection, annotator_assignment, learning_strategy, n_annotator, batch_size, metric):
    batch_size = n_classes_dict[dataset] * batch_size
    if rq == 'RQ1':
        fig = eval_RQ1(dataset, instance_selection, annotator_assignment, learning_strategy, n_annotator, batch_size, metric)
    elif rq == 'RQ2':
        fig = eval_RQ2(dataset, instance_selection, annotator_assignment, learning_strategy, n_annotator, batch_size,
                       metric)
    elif rq == 'RQ3':
        fig = eval_RQ3(dataset, instance_selection, annotator_assignment, learning_strategy, n_annotator, batch_size,
                       metric)
    elif rq == 'RQ4':
        fig = eval_RQ4(dataset, instance_selection, annotator_assignment, learning_strategy, n_annotator, batch_size,
                       metric)

    return fig


def eval_RQ1(
        dataset,
        instance_query_strategies,
        annotator_query_strategies,
        learning_strategies,
        n_annotator_list,
        batch_size,
        metric,
):
    fig = px.line()  # Initialize the figure
    for idx_i, instance_query_strategy in enumerate(instance_query_strategies):
        mean_instance_query = []
        std_instance_query = []
        for idx_a, annotator_query_strategy in enumerate(annotator_query_strategies):
            for idx_l, learning_strategy in enumerate(learning_strategies):
                for idx_n, n_annotator_per_instance in enumerate(n_annotator_list):
                    if learning_strategy in ['trace-reg', 'geo-reg-f',
                                             'geo-reg-w'] and annotator_query_strategy == 'intelligent':
                        annotator_query_strategy = learning_strategy
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

        # Add the line plot for this strategy
        fig.add_scatter(x=np.arange(init_batch_size, len(mean_i) * batch_size + init_batch_size, batch_size),
                        y=mean_i,
                        mode='lines',
                        name=f"({np.mean(mean_i):.4f}) {instance_query_strategy}",
                        line=dict(width=2),
                        opacity=0.7)

    fig.update_layout(
        xaxis_title='# Annotations queried',
        yaxis_title='Erroneous Annotation Rate' if metric == 'error_annotation_rate' else 'Misclassification Rate',
        legend_x=1,
        legend_y=1,
        margin=dict(t=50, b=50)
    )
    return fig


def eval_RQ2(
        dataset,
        instance_query_strategies,
        annotator_query_strategies,
        learning_strategies,
        n_annotator_list,
        batch_size,
        metric,
):
    fig = px.line()  # Initialize the figure
    for idx_l, learning_strategy in enumerate(learning_strategies):
        mean_learning_strategy = []
        std_learning_strategy = []
        for idx_i, instance_query_strategy in enumerate(instance_query_strategies):
            for idx_a, annotator_query_strategy in enumerate(annotator_query_strategies):
                for idx_n, n_annotator_per_instance in enumerate(n_annotator_list):
                    if learning_strategy in ['trace-reg', 'geo-reg-f',
                                             'geo-reg-w'] and annotator_query_strategy == 'intelligent':
                        annotator_query_strategy = learning_strategy
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

        # Add the line plot for this strategy
        fig.add_scatter(x=np.arange(init_batch_size, len(mean_a) * batch_size + init_batch_size, batch_size),
                        y=mean_a,
                        mode='lines',
                        name=f"({np.mean(mean_a):.4f}) {learning_strategy}",
                        line=dict(width=2),
                        opacity=0.7)

    fig.update_layout(
        xaxis_title='# Annotations queried',
        yaxis_title='Erroneous Annotation Rate' if metric == 'error_annotation_rate' else 'Misclassification Rate',
        legend_x=1,
        legend_y=1,
        margin=dict(t=50, b=50)
    )
    return fig


def eval_RQ3(
        dataset,
        instance_query_strategies,
        annotator_query_strategies,
        learning_strategies,
        n_annotator_list,
        batch_size,
        metric,
):
    fig = px.line()  # Initialize the figure
    for idx_l, learning_strategy in enumerate(learning_strategies):
        for idx_a, annotator_query_strategy in enumerate(annotator_query_strategies):
            for idx_i, instance_query_strategy in enumerate(instance_query_strategies):
                for idx_n, n_annotator_per_instance in enumerate(n_annotator_list):
                    if (annotator_query_strategy == 'intelligent' and
                            learning_strategy != 'majority-vote'):
                        annotator_query_strategy = learning_strategy

                    # Fetch the mean and std values along with the label
                    metric_mean, metric_std, label = get_metric(dataset, instance_query_strategy,
                                                                annotator_query_strategy,
                                                                learning_strategy, n_annotator_per_instance, batch_size,
                                                                metric)
                    idx_linestyle = idx_linestyle_dict[annotator_query_strategy]

                    if annotator_query_strategy in ['trace-reg', 'geo-reg-f', 'geo-reg-w']:
                        annotator_query_strategy = 'intelligent'

                    # Add the line plot for this strategy
                    fig.add_scatter(x=np.arange(batch_size, (len(metric_mean) + 1) * batch_size, batch_size),
                                    y=metric_mean,
                                    mode='lines',
                                    name=f"({np.mean(metric_mean):.4f}) {learning_strategy} - {annotator_query_strategy}",
                                    line=dict(color=colors[idx_l], dash=linestyles[idx_linestyle]),
                                    opacity=0.7)

    fig.update_layout(
        xaxis_title='# Annotations queried',
        yaxis_title='Erroneous Annotation Rate' if metric == 'error_annotation_rate' else 'Misclassification Rate',
        legend_x=1,
        legend_y=1,
        margin=dict(t=50, b=50)
    )
    return fig


def eval_RQ4(
        dataset,
        instance_query_strategies,
        annotator_query_strategies,
        learning_strategies,
        n_annotator_list,
        batch_size,
        metric,
):
    fig = px.line()  # Initialize the figure
    for idx_n, n_annotator_per_instance in enumerate(n_annotator_list):
        mean_n_annotator = []
        std_n_annotator = []
        for idx_i, instance_query_strategy in enumerate(instance_query_strategies):
            for idx_a, annotator_query_strategy in enumerate(annotator_query_strategies):
                for idx_l, learning_strategy in enumerate(learning_strategies):
                    if learning_strategy in ['trace-reg', 'geo-reg-f',
                                             'geo-reg-w'] and annotator_query_strategy == 'intelligent':
                        annotator_query_strategy = learning_strategy

                    # Get the data
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
        init_batch_size = get_init_batch_size(dataset, batch_size)

        # Add the line plot for this strategy
        fig.add_scatter(x=np.arange(init_batch_size, len(mean_a) * batch_size + init_batch_size, batch_size),
                        y=mean_a,
                        mode='lines',
                        name=f"({np.mean(mean_a):.4f}) {n_annotator_per_instance}",
                        line=dict(width=2),
                        opacity=0.7)

    fig.update_layout(
        xaxis_title='# Annotations queried',
        yaxis_title='Erroneous Annotation Rate' if metric == 'error_annotation_rate' else 'Misclassification Rate',
        legend_x=1,
        legend_y=1,
        margin=dict(t=50, b=50)
    )
    return fig


def plot_heatmap(
        dataset,
        heat_map_numpy,
        heat_map_sum,
        strategies,
):
    # Create the heatmap for the main heatmap_numpy
    fig = go.Figure(data=go.Heatmap(
        z=heat_map_numpy,
        x=strategies,
        y=strategies,
        colorscale="YlGnBu",
        zmin=0.0,
        zmax=1.0,
        text=np.round(heat_map_numpy, 2),
        hovertemplate='Strategy: %{x}<br>Value: %{z}<extra></extra>',
        showscale=False
    ))

    # Add text annotations for the heatmap
    for i in range(len(strategies)):
        for j in range(len(strategies)):
            fig.add_annotation(
                x=strategies[j], y=strategies[i],
                text=f"{heat_map_numpy[i, j]:.2f}",
                showarrow=False,
                font=dict(color=get_color(heat_map_numpy[i, j]))
            )

    # Create a secondary heatmap for heat_map_sum
    fig.add_trace(go.Heatmap(
        z=heat_map_sum,
        x=["average"],
        y=strategies,
        colorscale="YlGnBu",
        zmin=0.0,
        zmax=1.0,
        hovertemplate='Sum Value: %{z}<extra></extra>',
        colorbar=dict(len=0.5, thickness=20)
    ))

    # Add annotations for heat_map_sum
    for i in range(len(strategies)):
        fig.add_annotation(
            x="average", y=strategies[i],
            text=f"{heat_map_sum[i, 0]:.2f}",
            showarrow=False,
            font=dict(color=get_color(heat_map_sum[i, 0]))
        )

    # Update layout
    fig.update_layout(
        title=dataset,
        xaxis_nticks=len(strategies),
        yaxis_nticks=len(strategies),
        xaxis_title=None,
        yaxis_title=None
    )
    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
