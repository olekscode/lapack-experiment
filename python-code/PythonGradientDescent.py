import random


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

    def initialize_random_weights(self, size):
        self.weights = [random.random() for i in range(size)]

    def fit(self, input_matrix, actual_values):
        self.initialize_random_weights(len(input_matrix[0]))

        self.costHistory = []
        self.costHistory.append(self.cost_function(input_matrix, actual_values))

        self.performedIterations = 0

        while not self.has_converged() and self.performedIterations <= self.maxIterations:
            self.update_weights(input_matrix, actual_values)
            self.costHistory.append(self.cost_function(input_matrix, actual_values))
            self.performedIterations += 1

    def cost_function(self, input_matrix, actual_values):
        predicted_values = self.hypothesis_function(input_matrix)
        squared_errors = [(actual - predicted) ** 2 for actual, predicted in zip(actual_values, predicted_values)]
        return sum(squared_errors) / len(squared_errors) / 2

    def has_converged(self):
        if len(self.costHistory) == 0:
            return False
        if len(self.costHistory) < 2:
            return False
        precision = 1e-10

        current_cost = self.costHistory[-1]
        previous_cost = self.costHistory[-2]

        difference = current_cost - previous_cost

        if difference > 0 or self.is_nan(difference):
            raise Exception("Model is starting to diverge")

        return (abs(current_cost) < precision) or (abs(difference) < precision)

    def update_weights(self, input_matrix, actual_values):
        predicted_output_vector = self.hypothesis_function(input_matrix)

        cost_derivative = self.cost_derivative(predicted_output_vector, actual_values)
        weight_derivative = self.weight_derivative(input_matrix, cost_derivative)

        bias_derivative = self.bias_derivative(cost_derivative)

        learning_rate = self.learningRate
        learning_rate_times_weight_derivative = [x * learning_rate for x in weight_derivative]

        self.weights = [w - lrtwd for w, lrtwd in zip(self.weights, learning_rate_times_weight_derivative)]
        self.bias = self.bias - (self.learningRate * bias_derivative)

    def cost_derivative(self, predicted_output_vector, target_output_vector):
        return [prediction - actual_value for prediction, actual_value in
                zip(predicted_output_vector, target_output_vector)]

    def bias_derivative(self, cost_derivative):
        return sum(cost_derivative) / len(cost_derivative)

    def weight_derivative(self, input_matrix, cost_derivative_vector):
        weight_derivative = []
        for i in range(len(input_matrix)):
            row_times_vector = [element * cost_derivative_vector[i] for element in input_matrix[i]]
            weight_derivative.append(row_times_vector)

        answer = []
        for index_column in range(len(weight_derivative[0])):
            derivative = 0
            for row in weight_derivative:
                derivative += row[index_column]
            answer.append(derivative)
        return [(element / len(cost_derivative_vector)) for element in answer]

    def predict(self, input_matrix):
        return self.hypothesis_function(input_matrix)

    def hypothesis_function(self, input_matrix):
        return self.weighted_sum(input_matrix)

    def weighted_sum(self, input_matrix):
        weighted_sum = []
        for row in input_matrix:
            lists_multiplication = [element * weight for element, weight in zip(row, self.weights)]
            weighted_sum.append(sum(lists_multiplication) + self.bias)
        return weighted_sum
