import numpy as np
import unittest

from sklearn.svm import SVR

from skactiveml.classifier import PWC
from skactiveml.pool import ALCE


class ALCETest(unittest.TestCase):

    def setUp(self):
        self.X_cand = np.zeros((100, 2))
        self.X = np.zeros((6, 2))
        self.y = [0, 1, 1, 0, 2, 1]
        self.classes = [0, 1, 2]
        self.cost_matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        self.regressor = SVR()
        self.pwc = PWC()

    # Test init parameters
    def test_init_param_base_regressor(self):
        alce = ALCE(base_regressor=self.pwc, cost_matrix=self.cost_matrix)
        self.assertRaises(TypeError, alce.query, self.X_cand, self.X, self.y)

    def test_init_param_cost_matrix(self):
        alce = ALCE(base_regressor=self.regressor, cost_matrix='A')
        self.assertRaises(ValueError, alce.query, self.X_cand, self.X, self.y)

    def test_init_param_embed_dim(self):
        alce = ALCE(base_regressor=self.regressor,
                    cost_matrix=self.cost_matrix, embed_dim=1.5)
        self.assertRaises(TypeError, alce.query, self.X_cand, self.X, self.y)

        alce = ALCE(base_regressor=self.regressor,
                    cost_matrix=self.cost_matrix, embed_dim=0)
        self.assertRaises(ValueError, alce.query, self.X_cand, self.X, self.y)

    def test_init_param_missing_label(self):
        alce = ALCE(self.regressor, self.cost_matrix, missing_label=[1, 2, 3])
        self.assertRaises(TypeError, alce.query, self.X_cand, self.X, self.y)

    def test_init_param_classes(self):
        alce = ALCE(self.regressor, self.cost_matrix, classes=[0, 1])
        self.assertRaises(ValueError, alce.query, self.X_cand, self.X, self.y)

    def test_init_param_mds_params(self):
        alce = ALCE(self.regressor, self.cost_matrix, mds_params=0)
        self.assertRaises(TypeError, alce.query, self.X_cand, self.X, self.y)

    def test_init_param_nn_params(self):
        alce = ALCE(self.regressor, self.cost_matrix, nn_params=0)
        self.assertRaises(TypeError, alce.query, self.X_cand, self.X, self.y)

    def test_init_param_random_state(self):
        alce = ALCE(self.regressor, self.cost_matrix)
        self.assertTrue(hasattr(alce, 'random_state'))
        self.assertRaises(ValueError, alce.query, self.X_cand, self.X, self.y)

    # Test query parameters
    def test_query_param_X_cand(self):
        alce = ALCE(self.regressor, self.cost_matrix)
        self.assertRaises(ValueError, alce.query, X_cand=[], X=[], y=[])
        self.assertRaises(ValueError, alce.query, X_cand=[], X=self.X,
                          y=self.y)

    def test_query_param_X(self):
        alce = ALCE(self.regressor, self.cost_matrix)
        self.assertRaises(ValueError, alce.query, X_cand=self.X_cand,
                          X=np.ones((5, 3)), y=self.y)

    def test_query_param_y(self):
        alce = ALCE(self.regressor, self.cost_matrix)
        self.assertRaises(ValueError, alce.query, X_cand=self.X_cand,
                          X=self.X, y=[0, 1, 4, 0, 2, 1])

    def test_query_param_batch_size(self):
        alce = ALCE(self.regressor, self.cost_matrix)
        self.assertRaises(TypeError, alce.query, self.X_cand, self.X, self.y,
                          batch_size=1.0)
        self.assertRaises(ValueError, alce.query, self.X_cand, self.X, self.y,
                          batch_size=0)

    def test_query_param_return_utilities(self):
        alce = ALCE(self.regressor, self.cost_matrix)
        self.assertRaises(TypeError, alce.query, X_cand=self.X_cand,
                          return_utilities=None)
        self.assertRaises(TypeError, alce.query, X_cand=self.X_cand,
                          return_utilities=[])
        self.assertRaises(TypeError, alce.query, X_cand=self.X_cand,
                          return_utilities=0)

    def test_query(self):
        alce = ALCE(base_regressor=self.regressor, cost_matrix=1-np.eye(2))
        query_indices = alce.query([[0], [100], [200]], [[0], [200]], [0, 1])
        np.testing.assert_array_equal(query_indices, [1])


if __name__ == '__main__':
    unittest.main()
