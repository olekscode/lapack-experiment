class LinearRegressionFixture:
    def __init__(self):
        self.bias = 5
        self.weights = [2]

        function = lambda x: self.weights[0] * x + self.bias

        self.inputMatrix = [[2], [3], [1], [5], [2], [6]]
        self.outputVector = [function(x[0]) for x in self.inputMatrix]