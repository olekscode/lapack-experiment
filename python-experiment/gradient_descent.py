class LinearRegressionGradientDescent:
    def __init__(self):
        self.learningRate = 0.01
        self.maxIterations = 5000
        self.costHistory = []
        self.performedIterations = 0

    def initializeWeightsWithZeros(self, size):
        self.weights = [0] * size

    def fit(self, inputMatrix, outputVector):
        cost = self.calculateCost(inputMatrix, outputVector)
        self.costHistory.add(cost)

        self.performedIterations = 0

        while(self.hasConverged() or self.performedIterations >= self.maxIterations):
            self.updateWeights(inputMatrix, outputVector)
            cost = self.calculateCost(inputMatrix, outputVector)
            self.costHistory.add(cost)
            self.performedIterations += 1

    def calculateCost(self, inputMatrix, outputVector):
        predictedValues = self.predict(inputMatrix)
        squaredErrors = [(outputVector[i] - predictedValues[i])**2 for i in range(len(outputVector))]
        return sum(squaredErrors) / len(squaredErrors)


