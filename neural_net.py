import numpy
import scipy.special


class NeuralNet:
    def __init__(self, input_nodes, output_nodes, hidden_nodes, learning_rate):
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes
        self.hidden_nodes = hidden_nodes
        self.learning_rate = learning_rate

        # each row in matrix consists of weights for
        # corresponding neuron in hidden layer
        self.weights_h_i = numpy.random.rand(self.hidden_nodes, self.input_nodes) - 0.5
        self.weights_o_h = numpy.random.rand(self.output_nodes, self.hidden_nodes) - 0.5

    def train(self, input_list, target_list):
        """
        This function updates weights for all layers

        :param input_list:
        :param target_list:
        """

        # converting lists into arrays
        # .T is for transpose
        input_arr = numpy.array(input_list, ndmin = 2).T
        target_arr = numpy.array(target_list, ndmin = 2).T

        # input and output for hidden layer
        hidden_inputs = numpy.dot(self.weights_h_i, input_arr)
        hidden_outputs = self.activation(hidden_inputs)

        # input and output for final layer
        final_inputs = numpy.dot(self.weights_o_h, hidden_outputs)
        final_outputs = self.activation(final_inputs)

        # errors of final layer
        output_errors = target_arr - final_outputs

        # errors of hidden layer
        hidden_errors = numpy.dot(self.weights_o_h.T, output_errors)

        # updating weights between hidden and final layers
        self.weights_o_h += self.learning_rate * numpy.dot(
            (output_errors * final_outputs * (1.0 - final_outputs)),
            numpy.transpose(hidden_outputs)
        )

        # updating weights between input and hidden layers
        self.weights_h_i += self.learning_rate * numpy.dot(
            (hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
            numpy.transpose(input_arr)
        )

    def query(self, input_list):
        """
        This function calculates inputs and outputs for layers

        :param input_list:
        :return: final_outputs
        """
        input_arr = numpy.array(input_list, ndmin = 2).T

        # input and output for hidden layer
        hidden_inputs = numpy.dot(self.weights_h_i, input_arr)
        hidden_outputs = self.activation(hidden_inputs)

        # input and output for final layer
        final_inputs = numpy.dot(self.weights_o_h, hidden_outputs)
        final_outputs = self.activation(final_inputs)

        return final_outputs

    def activation(self, input_arr):
        """
        This function counts new neuron signal
        using sigmoid function

        :param input_arr:
        """
        return scipy.special.expit(input_arr)
