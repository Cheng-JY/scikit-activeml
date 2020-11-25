import unittest
import inspect
from importlib import import_module
from os import path

import numpy as np
from sklearn.datasets import make_blobs

from skactiveml import pool
from skactiveml.classifier import PWC, CMM
from skactiveml.utils import is_unlabeled, MISSING_LABEL, \
    initialize_class_with_kwargs


class TestGeneral(unittest.TestCase):

    def setUp(self):
        self.MISSING_LABEL = MISSING_LABEL
        self.X, self.y_true = make_blobs(n_samples=10, n_features=2, centers=2,
                                         cluster_std=1, random_state=1)
        self.budget = 5

        self.query_strategies = {}
        for qs_name in pool.__all__:
            self.query_strategies[qs_name] = getattr(pool, qs_name)
        print(self.query_strategies.keys())

    def test_al_cycle(self):
        for init_budget in [5, 1, 0]:
            for qs_name in self.query_strategies:
                with self.subTest(msg="Basic AL Cycle",
                                  init_budget=init_budget, qs_name=qs_name):
                    y = np.full(self.y_true.shape, self.MISSING_LABEL)
                    y[0:init_budget] = self.y_true[0:init_budget]

                    qs, clf = self._initialize_query_strategy(qs_name)


                    for b in range(self.budget):
                        unlabeled = np.where(is_unlabeled(y))[0]
                        clf.fit(self.X, y)
                        unlabeled_id = qs.query(self.X[unlabeled], X=self.X,
                                                y=y, X_eval=self.X,
                                                sample_weight=np.ones(
                                                    len(unlabeled)))
                        sample_id = unlabeled[unlabeled_id]
                        y[sample_id] = self.y_true[sample_id]

    def test_param(self):
        for qs_name in self.query_strategies:
            with self.subTest(msg="Param Test", qs_name=qs_name):
                # get init_params
                qs_class = self.query_strategies[qs_name]
                init_params = list(inspect.signature(
                                    qs_class).parameters.keys())

                # check init params
                values = [Dummy() for i in range(len(init_params))]
                qs_obj = qs_class(*values)
                for param, value in zip(init_params, values):
                    self.assertTrue(
                        hasattr(qs_obj, param),
                        msg='"{}" not tested for __init__()'.format(param))
                    self.assertEqual(getattr(qs_obj, param), value)

                # get query_params
                qs, clf = self._initialize_query_strategy(qs_name)
                query_params = list(inspect.signature(
                    qs.query).parameters.keys())

                # get test class to check
                class_file_name = path.basename(inspect.getfile(qs_class))[:-3]
                mod = import_module(
                    'skactiveml.pool.tests.test' + class_file_name)
                self.assertTrue(hasattr(mod, 'Test' + qs_class.__name__),
                                msg='{} has no test called {}'.format(qs_name,
                                                                      'Test' + qs_class.__name__))
                test_obj = getattr(mod, 'Test' + qs_class.__name__)

                # check query params
                for param in query_params:
                    self.assertTrue(
                        hasattr(test_obj, 'test_query_param_' + param),
                        msg='"{}" param not tested for query()'.format(param))


    def _initialize_query_strategy(self, qs_name):
        if qs_name == "FourDS":
            clf = CMM(classes=np.unique(self.y_true),
                      missing_label=MISSING_LABEL)
        else:
            clf = PWC(classes=np.unique(self.y_true),
                      missing_label=MISSING_LABEL)

        qs = initialize_class_with_kwargs(
            self.query_strategies[qs_name],
            clf=clf, perf_est=clf, classes=np.unique(self.y_true),
            random_state=1)
        return qs, clf

class Dummy:
    def __init__(self):
        pass