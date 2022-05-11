class LinearRegressionGradientDescent:
    def __init__(self):
        self.learningRate = 0.01
        self.maxIterations = 5000
        self.costHistory = []
        self.performedIterations = 0
        self.weights = []
        self.bias = 0

    def isNaN(num):
        return num!= num

    def initializeWeightsWithZeros(self, size):
        self.weights = [0] * size

    def fit(self, inputMatrix, actualValues):
        self.initializeWeightsWithZeros(len(inputMatrix[0]))

        cost = self.costFunction(inputMatrix, actualValues)
        self.costHistory.append(cost)

        self.performedIterations = 0

        while(self.hasConverged() or self.performedIterations >= self.maxIterations):
            self.updateWeights(inputMatrix, actualValues)
            cost = self.costFunction(inputMatrix, actualValues)
            self.costHistory.add(cost)
            self.performedIterations += 1

    def costFunction(self, inputMatrix, actualValues):
        predictedValues = self.predict(inputMatrix)
        squaredErrors = [(actualValues[i] - predictedValues[i])**2 for i in range(len(actualValues))]
        return sum(squaredErrors) / len(squaredErrors)

    def hasConverged(self):
        if len(self.costHistory) == 0:
            return False
        if len(self.costHistory) < 2:
            return False
        precision = 1e-10
        difference = self.costHistory[len(self.costHistory) - 1] - self.costHistory[len(self.costHistory) - 2]

        if difference > 0 or self.isNan(difference):
            raise Exception("Model is starting to diverge")

        return (abs(self.costHistory[len(self.costHistory) - 1] - self.costHistory[len(self.costHistory) - 2]) < precision) or (difference < precision)

    def updateWeights(self, inputMatrix, actualValues):
        predictedOutputVector = self.hypothesisFunction(inputMatrix)

        costDerivative = self.costDerivative(predictedOutputVector, actualValues)
        weightDerivative = self.weightDerivative(inputMatrix, actualValues)

        biasDerivative = self.biasDerivative(costDerivative)

        self.weights = self.weights - (self.learningRate * weightDerivative)
        self.bias = self.bias - (self.learningRate * biasDerivative)

    def costDerivative(self, predictedOutputVector, targetOutputVector):
        return 2 * (predictedOutputVector - targetOutputVector)

    def biasDerivative(self, costDerivative):
        return sum(costDerivative) / len(costDerivative)

    def weightDerivative(self, inputMatrix, costDerivativeVector):
        weightDerivative = []
        for i in range(inputMatrix):
            weightDerivative.append(inputMatrix[i] * costDerivativeVector[i])
        return sum(weightDerivative) / len(weightDerivative)

    def predict(self, inputMatrix):
        return self.hypothesisFunction(inputMatrix)

    def hypothesisFunction(self, inputMatrix):
        return self.weightedSum(inputMatrix)

    def weightedSum(self, inputMatrix):
        weightedSum = []
        for i in range(len(inputMatrix)):
            listsMultiplication = [inputMatrix[i][j] * self.weights[j] for j in range(len(self.weights))]
            weightedSum.append(sum(listsMultiplication) + self.bias)
        return weightedSum
    