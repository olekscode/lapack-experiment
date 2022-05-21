import unittest
from PythonGradientDescent import PythonGradientDescent
from Fixtures import LinearRegressionTrivialFixture


class LinearRegressionTest(unittest.TestCase):

    def regression(self):
        return PythonGradientDescent()

    def setUp(self):
        super().setUp()
        self.fixture = LinearRegressionTrivialFixture()

        self.model = self.regression()
        self.model.learningRate = self.fixture.learning_rate
        self.model.bias = self.fixture.initial_bias
        self.model.weights = self.fixture.initial_weights

    def test_prediction_before_learning(self):
        prediction = self.model.predict(self.fixture.input_matrix)
        self.assertEqual(prediction, self.fixture.expected_prediction_before_learning)

    def test_expected_weights_at_first_iteration(self):
        self.model.update_weights(self.fixture.input_matrix, self.fixture.output_vector)
        self.assertEqual(self.fixture.expected_weights_at_first_iteration, self.model.weights)

    def test_expected_weight_derivative_at_first_iteration(self):
        predicted_output_vector = self.model.predict(self.fixture.input_matrix)
        cost_derivative = self.model.cost_derivative(predicted_output_vector, self.fixture.output_vector)
        weight_derivative = self.model.weight_derivative(self.fixture.input_matrix, cost_derivative)

        self.assertEqual(weight_derivative, self.fixture.expectedWeightDerivativeAtFirstIteration)

    def test_expected_bias_derivative_at_first_iteration(self):
        predicted_output_vector = self.model.predict(self.fixture.input_matrix)
        cost_derivative = self.model.cost_derivative(predicted_output_vector, self.fixture.output_vector)
        bias_derivative = self.model.bias_derivative(cost_derivative)

        self.assertEqual(bias_derivative, self.fixture.expectedBiasDerivativeAtFirstIteration)

    def test_expected_bias_at_first_iteration(self):
        self.model.update_weights(self.fixture.input_matrix, self.fixture.output_vector)
        self.assertEqual(self.model.bias, self.fixture.expectedBiasAtFirstIteration)

    def test_cost_before_learning(self):
        cost = self.model.cost_function(self.fixture.input_matrix, self.fixture.output_vector)
        self.assertEqual(cost, self.fixture.expectedCostBeforeLearning)