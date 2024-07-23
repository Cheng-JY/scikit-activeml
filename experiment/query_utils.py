import numpy as np

from skactiveml.pool import UncertaintySampling, RandomSampling, CoreSet


def gen_seed(random_state:np.random.RandomState):
    return random_state.randint(0, 2**31)


def gen_random_state(random_state:np.random.RandomState):
    return np.random.RandomState(gen_seed(random_state))


def create_instance_query_strategy(name, random_state, missing_label):
    query_strategy_factory_functions = {
        'random': lambda random_state: RandomSampling(random_state=gen_seed(random_state), missing_label=missing_label),
        'uncertainty': lambda random_state: UncertaintySampling(random_state=gen_seed(random_state), missing_label=missing_label),
        'coreset': lambda random_state: CoreSet(random_state=gen_seed(random_state), missing_label=missing_label)
    }
    return query_strategy_factory_functions[name](random_state)


def get_annotator_performance(name, shape):
    A_perf_dic = {
        'random': np.ones(shape=shape),
        'round-robin': np.zeros(shape=shape),
        'trace-reg': np.ones(shape=shape[1]),
        'geo-reg-f': np.ones(shape=shape[1]),
        'geo-reg-w': np.ones(shape=shape[1]),
    }
    return A_perf_dic.get(name, np.ones(shape=shape))