import unittest
from gradient_descent import LinearRegressionGradientDescent
from fixtures import LinearRegressionFixture


class LinearRegressionTest(unittest.TestCase):

    def regression(self):
        return LinearRegressionGradientDescent()

    def setUp(self):
        super().setUp()
        self.model = self.regression()
        self.fixture = LinearRegressionFixture()

    def test_emptyInputMatrix(self):
        input = [[]]
        output = [1]

        with self.assertRaises(Exception):
            self.model.fit(input, output)

    def test_emptyOutputVector(self):
        input = [[5], [3]]
        output = []

        with self.assertRaises(Exception):
            self.model.fit(input, output)

    def test_fitInputWithInconsistentTypes(self):
        input = [['number'], [3], [1]]
        output = [1, 3, 1]

        with self.assertRaises(Exception):
            self.model.fit(input, output)

    def test_fitSizeMismatch(self):
        input = [[2], [3]]
        output = [1]

        with self.assertRaises(Exception):
            self.model.fit(input, output)

    def test_fitOutputWithInconsistentTypes(self):
        input = [[1], [3], [1]]
        output = ['number', 3, 1]

        with self.assertRaises(Exception):
            self.model.fit(input, output)

    def test_initializeWeightsWithZeros(self):
        expectedWeights = [0, 0, 0]
        self.model.initializeWeightsWithZeros(3)
        self.assertEqual(self.model.weights, expectedWeights)

    def test_learningRateIsInitialized(self):
        self.assertTrue(self.model.learningRate > 0)

    def test_maxIterationsIsInitialized(self):
        self.assertTrue(self.model.maxIterations > 0)

    def test_biasAlmostEqual(self):
        self.model.fit(self.fixture.inputMatrix, self.fixture.outputVector)
        self.assertAlmostEqual(self.model.bias, self.fixture.bias)


if __name__ == '__main__':
    unittest.main()