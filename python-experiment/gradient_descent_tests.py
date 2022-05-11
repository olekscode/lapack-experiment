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

    def test_exactFitSingleVariable(self):
        newInput = [ [4], [1], [7], [0] ]
        expectedOutput = [ 11, 5, 17, 3 ]

        self.model.learningRate = 0.01
        self.model.maxIterations = 3000

        self.model.fit(self.fixture.inputMatrix, self.fixture.outputVector)
        actualOutput = self.model.predict(newInput)
        print(actualOutput)

        for i in range(len(actualOutput)):
            self.assertAlmostEquals(actualOutput[i], expectedOutput[i])

    def test_weightsAreAlmostEqual(self):
        self.model.learningRate = 0.01
        self.model.maxIterations = 3000

        self.model.fit(self.fixture.inputMatrix, self.fixture.outputVector)

        for i in len(self.fixture.weights):
            self.assertAlmostEquals(self.model.weights[i], self.fixture.weights[i])

    def test_divergingExecption(self):
        input = [ [ 13421525235235235235 ], [ 3 ], [ 0.1 ], [ 0.000005 ], [ 241241241124124124 ], [ 6412412412414 ], [ 45345 ], [ 5 ], [ 53 ], [ 5 ], [ 3 ], [ 1 ], [ 2 ], [ 1 ], [ 0.09 ], [ 0.4 ], [ 0.0009 ], [ 5 ], [ 234242342423423 ], [ 0.9888 ], [ 0.0000009 ] ]
        output = [ 4, 234, 523, 523, 5, 63456346346346, 636463, 63463, 0.253, 0.84234, 0.00042, 243, 4, 2, 2, 5, 2, 5235235, 0.0005, 3, 3 ]

        self.model.learningRate = 100
        self.model.maxIterations = 3000

        with self.assertRaises(Exception):
            self.model.fit(input, output)

if __name__ == '__main__':
    unittest.main()