import unittest
from PythonGradientDescent import PythonGradientDescent
from Fixtures import LinearRegressionFixture


class LinearRegressionTest(unittest.TestCase):

    def regression(self):
        return PythonGradientDescent()

    def setUp(self):
        super().setUp()
        self.fixture = LinearRegressionFixture()
        self.model = self.regression()
        self.model.learningRate = 0.1
        self.model.maxIterations = 3000

    def test_fitSizeMismatch(self):
        input_matrix = [[2], [3]]
        output = [1]

        with self.assertRaises(Exception):
            self.model.fit(input_matrix, output)

    def test_initializeWeightsWithZeros(self):
        expected_weights = [0, 0, 0]
        self.model.initialize_weights_with_zeros(3)
        self.assertEqual(self.model.weights, expected_weights)

    def test_learningRateIsInitialized(self):
        self.assertTrue(self.model.learningRate > 0)

    def test_maxIterationsIsInitialized(self):
        self.assertTrue(self.model.maxIterations > 0)

    def test_biasAlmostEqual(self):
        self.model.fit(self.fixture.inputMatrix, self.fixture.outputVector)
        self.assertAlmostEqual(self.model.bias, self.fixture.bias, places=3)

    def test_exactFitSingleVariable(self):
        new_input = [[4], [1], [7], [0]]
        expected_output = [11, 5, 17, 3]

        self.model.fit(self.fixture.inputMatrix, self.fixture.outputVector)
        actual_output = self.model.predict(new_input)

        for i in range(len(actual_output)):
            self.assertAlmostEquals(actual_output[i], expected_output[i], places=3)

    def test_weightsAreAlmostEqual(self):
        self.model.fit(self.fixture.inputMatrix, self.fixture.outputVector)

        for i in range(len(self.fixture.weights)):
            self.assertAlmostEquals(self.model.weights[i], self.fixture.weights[i], places=3)

    def test_divergingException(self):
        input_matrix = [[13421525235235235235], [3], [0.1], [0.000005],
                        [241241241124124124], [6412412412414], [45345], [5], [53],
                        [5], [3], [1], [2], [1], [0.09], [0.4], [0.0009], [5],
                        [234242342423423], [0.9888], [0.0000009]]
        output = [4, 234, 523, 523, 5, 63456346346346, 636463, 63463, 0.253,
                  0.84234, 0.00042, 243, 4, 2, 2, 5, 2, 5235235, 0.0005, 3, 3]

        self.model.learningRate = 100

        with self.assertRaises(Exception):
            self.model.fit(input_matrix, output)

    def test_weightDerivative(self):
        result = self.model.weight_derivative([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], [9, 8, 7, 6])
        self.assertEqual(result, [37.5, 45, 52.5])

    def test_costDerivative(self):
        result = self.model.cost_derivative([10, 20, 30], [11, 22, 33])
        self.assertEqual(result, [-1, -2, -3])

        result = self.model.cost_derivative([11, 22, 33], [10, 20, 30])
        self.assertEqual(result, [1, 2, 3])

    def test_biasDerivative(self):
        result = self.model.bias_derivative([3, 4, 5, 6])
        self.assertEqual(result, 4.5)

    def test_weightedSum(self):
        self.model.bias = 2
        self.model.weights = [4, 5, 6]
        result = self.model.weighted_sum([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        self.assertEqual(result, [34, 79, 124])


if __name__ == '__main__':
    unittest.main()
