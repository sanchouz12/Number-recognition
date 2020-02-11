import numpy
import scipy.special


class NeuralNet:
    def __init__(self, input_nodes, output_nodes, nodes_arr, learning_rate):
        """
        :param input_nodes: amount of nodes of input layer
        :param output_nodes: amount of nodes of output layer
        :param nodes_arr: list of nodes of hidden layers
        :param learning_rate: coefficient of learning
        """
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes
        self.nodes_arr = nodes_arr
        self.learning_rate = learning_rate

        # each row in matrix consists of weights for
        # corresponding neuron in hidden layer
        self.weights = []

        self.weights.append(numpy.random.rand(self.nodes_arr[0], self.input_nodes) - 0.5)
        for index in range(1, len(self.nodes_arr)):
            curr_nodes = self.nodes_arr[index]
            prev_nodes = self.nodes_arr[index - 1]

            self.weights.append(numpy.random.rand(curr_nodes, prev_nodes) - 0.5)

        last_nodes = self.nodes_arr[-1]

        self.weights.append(numpy.random.rand(self.output_nodes, last_nodes) - 0.5)

        self.outputs = []
        self.errors = []

    def train(self, input_list, target_list):
        """
        This function calculates errors, based on target value, for all layers.
        Using update_weights function all weights are being updated.

        :param input_list: list of inputs
        :param target_list: list of expected values
        """
        # converting list into a numpy array
        # .T is for transpose
        input_arr = numpy.array(input_list, ndmin = 2).T
        target_arr = numpy.array(target_list, ndmin = 2).T

        self.query(input_arr)

        self.count_errors(target_arr)

        self.update_weights()

        self.outputs = []
        self.errors = []

    def query(self, input_arr):
        """
        This function calculates inputs and outputs for all layers.

        :param input_arr: numpy array of inputs for a neural net
        :return final_outputs: numpy array of outputs of the output layer
        """
        self.outputs.append(input_arr)

        first_hidden_inputs = numpy.dot(self.weights[0], input_arr)
        first_hidden_outputs = self.activation(first_hidden_inputs)

        self.outputs.append(first_hidden_outputs)

        # last is not included
        for index in range(2, len(self.weights)):
            weights_arr = self.weights[index - 1]
            prev_outputs = self.outputs[index - 1]

            curr_inputs = numpy.dot(weights_arr, prev_outputs)
            curr_outputs = self.activation(curr_inputs)

            self.outputs.append(curr_outputs)

        final_inputs = numpy.dot(self.weights[-1], self.outputs[-1])
        final_outputs = self.activation(final_inputs)

        self.outputs.append(final_outputs)

        return final_outputs

    def activation(self, input_arr):
        """
        This function counts new neuron signal
        using sigmoid function

        :param input_arr: numpy array of inputs for a layer
        :return: numpy array
        """
        return scipy.special.expit(input_arr)

    def count_errors(self, target_arr):
        """
        This function calculates errors for all layers.

        :param target_arr: numpy array of expected values
        """
        self.errors.append(target_arr - self.outputs[-1])
        for index in range(len(self.weights) - 2, -1, -1):
            prev_errors = self.errors[0]

            self.errors.insert(0, numpy.dot(self.weights[index + 1].T, prev_errors))

    def update_weights(self):
        """
        This function updates all weights.
        """
        self.weights[0] += self.learning_rate * numpy.dot(
            (self.errors[0] * self.outputs[1] * (1.0 - self.outputs[1])),
            numpy.transpose(self.outputs[0])
        )
        for index in range(1, len(self.weights)):
            curr_errors = self.errors[index]
            curr_output = self.outputs[index + 1]
            prev_output = self.outputs[index]

            self.weights[index] += self.learning_rate * numpy.dot(
                (curr_errors * curr_output * (1.0 - curr_output)),
                numpy.transpose(prev_output)
            )
