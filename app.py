import numpy

from neural_net import NeuralNet


class App:
    def __init__(self):
        self.layers = 0
        self.neurons = []
        self.coefficient = 0
        self.net = NeuralNet(0, 0, [0], 0)

    def menu(self):
        option = input("Input 1 to load the net's weights from file\n"
                       "Input 2 to train new net\n")

        if option == "1":
            self.load()
            print("Data loaded, starting to train...")
        elif option == "2":
            self.get_params()
            self.train()
            print("The net has been trained, starting to test...")

        self.test()

        option = input("Would you like to save the weights (y/n)?\n")

        if option == "y":
            self.save()

    def get_params(self):
        self.layers = int(input("How many hidden layers will the neural net contain?\n"))

        for index in range(self.layers):
            self.neurons.append(int(input(
                "How many neurons will be in {} layer?\n".format(index + 1)
            )))

        self.coefficient = float(input("What's the training coefficient?\n"))

    def train(self):
        self.net = NeuralNet(784, 10, self.neurons, self.coefficient)

        with open("mnist_dataset/mnist_train.csv", "r") as file:
            train_data = file.readlines()

        epochs = int(input("How many epochs of training the net should be?\n"))

        # training the net
        for x in range(epochs):
            for line in train_data:
                line_data = line.split(",")
                marker = line_data[0]

                # scaling input data
                inputs = (numpy.asfarray(line_data[1:]) / 255.0 * 0.99) + 0.01

                # there are 10 numbers, we try to recognize only 1,
                # so only one number in targets array will be 0.99
                targets = numpy.zeros(10) + 0.01
                targets[int(marker)] = 0.99

                self.net.train(inputs, targets)

    def test(self):
        with open("mnist_dataset/mnist_test.csv", "r") as file:
            test_data = file.readlines()

        # collecting scores
        scorecard = []

        # testing the net
        for line in test_data:
            line_data = line.split(",")
            marker = line_data[0]

            inputs = (numpy.asfarray(line_data[1:]) / 255.0 * 0.99) + 0.01
            outputs = self.net.query(inputs)

            label = numpy.argmax(outputs)

            print("Expected output - {}, received - {}".format(marker, label))

            if int(marker) == label:
                scorecard.append(1)
            else:
                scorecard.append(0)

        scorecard_arr = numpy.asarray(scorecard)
        precision = scorecard_arr.sum() / scorecard_arr.size

        print("Precision:", precision)

    def load(self):
        with open("weights.txt", "r") as file:
            data = file.readlines()

        self.net.weights = []
        payload = []

        for line in data:
            if line == "\n":
                weight = numpy.array(payload)

                self.net.weights.append(weight)
                payload = []
            else:
                arr = line.split(" ")
                arr = numpy.asfarray(arr)

                payload.append(arr)
        
        with open("params.txt", "r") as file:
            data = file.readlines()
        
        self.neurons = data[0]
        self.coefficient = data[1]

    def save(self):
        with open("weights.txt", "w") as file:
            file.writelines("")
        with open("params.txt", "w") as file:
            file.writelines("")

        with open("weights.txt", "a") as file:
            for arr in self.net.weights:
                numpy.savetxt(file, arr)
                file.writelines("\n")

        with open("params.txt", "a") as file:
            file.writelines(f"{self.neurons}\n")
            file.writelines(f"{self.coefficient}")


if __name__ == "__main__":
    app = App()

    app.menu()
