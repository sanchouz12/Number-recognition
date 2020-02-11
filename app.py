import numpy

from neural_net import NeuralNet

layers = int(input("How many hidden layers will the neural net contain?\n"))
neurons = []

for index in range(layers):
    neurons.append(int(input(
        "How many neurons will be in {} layer?\n".format(index + 1)
    )))

coefficient = float(input("What's the training coefficient?\n"))

net = NeuralNet(784, 10, neurons, coefficient)

with open("mnist_dataset/mnist_test.csv", "r") as file:
    test_data = file.readlines()

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

        net.train(inputs, targets)

# collecting scores
scorecard = []

# testing the net
for line in test_data:
    line_data = line.split(",")
    marker = line_data[0]

    inputs = (numpy.asfarray(line_data[1:]) / 255.0 * 0.99) + 0.01
    outputs = net.query(inputs)

    label = numpy.argmax(outputs)

    print("Expected output - {}, received - {}".format(marker, label))

    if int(marker) == label:
        scorecard.append(1)
    else:
        scorecard.append(0)

scorecard_arr = numpy.asarray(scorecard)
precision = scorecard_arr.sum() / scorecard_arr.size

print("Precision:", precision)
