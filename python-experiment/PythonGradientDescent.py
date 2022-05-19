class PythonGradientDescent:
    def __init__(self):
        self.learningRate = 0.01
        self.maxIterations = 5000
        self.costHistory = []
        self.performedIterations = 0
        self.weights = []
        self.bias = 0

    def isNaN(self, num):
        return num!= num

    def initializeWeightsWithZeros(self, size):
        self.weights = [0] * size

    def fit(self, inputMatrix, actualValues):
        self.initializeWeightsWithZeros(len(inputMatrix[0]))

        self.costHistory = []
        self.costHistory.append(self.costFunction(inputMatrix, actualValues))

        self.performedIterations = 0

        while(not self.hasConverged() and self.performedIterations <= self.maxIterations):
            self.updateWeights(inputMatrix, actualValues)
            self.costHistory.append(self.costFunction(inputMatrix, actualValues))
            self.performedIterations += 1

    def costFunction(self, inputMatrix, actualValues):
        predictedValues = self.hypothesisFunction(inputMatrix)
        squaredErrors = [ (actualValues[i] - predictedValues[i])**2 for i in range(len(actualValues)) ]
        return sum(squaredErrors) / len(squaredErrors)

    def hasConverged(self):
        if len(self.costHistory) == 0:
            return False
        if len(self.costHistory) < 2:
            return False
        precision = 1e-10
        difference = self.costHistory[len(self.costHistory) - 1 ] - self.costHistory[len(self.costHistory) - 2]

        if difference > 0 or self.isNan(difference):
            raise Exception("Model is starting to diverge")

        return (abs(self.costHistory[len(self.costHistory) - 1]) < precision) or (difference < precision)

    def updateWeights(self, inputMatrix, actualValues):
        predictedOutputVector = self.hypothesisFunction(inputMatrix)

        costDerivative = self.costDerivative(predictedOutputVector, actualValues)
        weightDerivative = self.weightDerivative(inputMatrix, actualValues)

        biasDerivative = self.biasDerivative(costDerivative)

        learningRateTimesWeightDerivative = [ x * self.learningRate for x in weightDerivative ]
        self.weights = [ (self.weights[i] - learningRateTimesWeightDerivative[i]) for i in range(len(self.weights)) ]
        self.bias = self.bias - (self.learningRate * biasDerivative)

    def costDerivative(self, predictedOutputVector, targetOutputVector):
        subtracted = list()
        for i in range(len(predictedOutputVector)):
            item = 2 * (predictedOutputVector[i] - targetOutputVector[i])
            subtracted.append(item)
        return subtracted

    def biasDerivative(self, costDerivative):
        return sum(costDerivative) / len(costDerivative)

    def weightDerivative(self, inputMatrix, costDerivativeVector):
        weightDerivative = []
        for i in range(len(inputMatrix)):
            rowTimesVector = [ inputMatrix[i][j] * costDerivativeVector[i] for j in range(len(inputMatrix[i])) ]
            weightDerivative.append(rowTimesVector)

        answer = []
        for column in range(len(weightDerivative[0])):
            sum = 0
            for row in weightDerivative:
                sum += row[column]
            answer.append(sum)
        return [ (element / len(costDerivativeVector)) for element in answer ]

    def predict(self, inputMatrix):
        return self.hypothesisFunction(inputMatrix)

    def hypothesisFunction(self, inputMatrix):
        return self.weightedSum(inputMatrix)

    def weightedSum(self, inputMatrix):
        weightedSum = []
        for i in range(len(inputMatrix)):
            listsMultiplication = [ inputMatrix[i][j] * self.weights[j] for j in range(len(self.weights)) ]
            weightedSum.append(sum(listsMultiplication) + self.bias)
        return weightedSum
