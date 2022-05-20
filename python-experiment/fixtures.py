class LinearRegressionFixture:
    def __init__(self):
        self.bias = 3
        self.weights = [2]

        function = lambda x: self.weights[0] * x + self.bias

        self.inputMatrix = [[2], [3], [1], [5], [2], [6]]
        self.outputVector = [function(x[0]) for x in self.inputMatrix]


class LinearRegressionTrivialFixture:
    def __init__(self):
        self.initial_bias = 0
        self.initial_weights = [0, 0]
        self.input_matrix = [[1, 2], [3, 4], [5, 6]]
        self.output_vector = [1, 0, 1]
        self.learning_rate = 0.1

        self.expected_prediction_before_learning = [0, 0, 0]
        self.expectedCostBeforeLearning = 1 / 3
        self.expectedWeightDerivativeAtFirstIteration = [-2, -8 / 3]
        self.expectedBiasDerivativeAtFirstIteration = -2 / 3
        self.expected_weights_at_first_iteration = [0.2, 4 / 15]
        self.expectedBiasAtFirstIteration = 1 / 15
