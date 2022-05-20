class PythonGradientDescent:
    def __init__(self):
        self.learningRate = 0.01
        self.maxIterations = 5000
        self.costHistory = []
        self.performedIterations = 0
        self.weights = []
        self.bias = 0

    def is_nan(self, num):
        return num != num

    def initializeWeightsWithZeros(self, size):
        self.weights = [0] * size

    def fit(self, inputMatrix, actualValues):
        self.initializeWeightsWithZeros(len(inputMatrix[0]))

        self.costHistory = []
        self.costHistory.append(self.cost_function(inputMatrix, actualValues))

        self.performedIterations = 0

        while not self.has_converged() and self.performedIterations <= self.maxIterations:
            self.update_weights(inputMatrix, actualValues)
            self.costHistory.append(self.cost_function(inputMatrix, actualValues))
            self.performedIterations += 1

    def cost_function(self, inputMatrix, actualValues):
        predictedValues = self.hypothesis_function(inputMatrix)
        squaredErrors = [(actualValues[i] - predictedValues[i]) ** 2 for i in range(len(actualValues))]
        return sum(squaredErrors) / len(squaredErrors)

    def has_converged(self):
        if len(self.costHistory) == 0:
            return False
        if len(self.costHistory) < 2:
            return False
        precision = 1e-10
        difference = self.costHistory[len(self.costHistory) - 1] - self.costHistory[len(self.costHistory) - 2]

        if difference > 0 or self.isNan(difference):
            raise Exception("Model is starting to diverge")

        return (abs(self.costHistory[len(self.costHistory) - 1]) < precision) or (difference < precision)

    def update_weights(self, inputMatrix, actualValues):
        predictedOutputVector = self.hypothesis_function(inputMatrix)

        costDerivative = self.cost_derivative(predictedOutputVector, actualValues)
        weightDerivative = self.weight_derivative(inputMatrix, actualValues)

        biasDerivative = self.bias_derivative(costDerivative)

        learningRateTimesWeightDerivative = [x * self.learningRate for x in weightDerivative]
        self.weights = [(self.weights[i] - learningRateTimesWeightDerivative[i]) for i in range(len(self.weights))]
        self.bias = self.bias - (self.learningRate * biasDerivative)

    def cost_derivative(self, predictedOutputVector, targetOutputVector):
        subtracted = list()
        for i in range(len(predictedOutputVector)):
            item = 2 * (predictedOutputVector[i] - targetOutputVector[i])
            subtracted.append(item)
        return subtracted

    def bias_derivative(self, costDerivative):
        return sum(costDerivative) / len(costDerivative)

    def weight_derivative(self, inputMatrix, costDerivativeVector):
        weightDerivative = []
        for i in range(len(inputMatrix)):
            rowTimesVector = [inputMatrix[i][j] * costDerivativeVector[i] for j in range(len(inputMatrix[i]))]
            weightDerivative.append(rowTimesVector)

        answer = []
        for column in range(len(weightDerivative[0])):
            sum = 0
            for row in weightDerivative:
                sum += row[column]
            answer.append(sum)
        return [(element / len(costDerivativeVector)) for element in answer]

    def predict(self, inputMatrix):
        return self.hypothesis_function(inputMatrix)

    def hypothesis_function(self, inputMatrix):
        return self.weighted_sum(inputMatrix)

    def weighted_sum(self, inputMatrix):
        weightedSum = []
        for i in range(len(inputMatrix)):
            listsMultiplication = [inputMatrix[i][j] * self.weights[j] for j in range(len(self.weights))]
            weightedSum.append(sum(listsMultiplication) + self.bias)
        return weightedSum
