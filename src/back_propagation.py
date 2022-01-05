# -*- encoding: utf-8 -*-
#Date: 05/Jan/2022
#Author: Steven Huang, Auckland, NZ
#License: MIT License
"""
Description: Backpropagation step by step
1.Initialize Network.
2.Forward Propagate.
3.Back Propagate Error.
4.Train Network.
5.Predict.
"""

import numpy as np

np.random.seed(10) #keep random always same

def initialize_network(n_inputs, n_hidden, n_outputs):
    network = []
    hidden_layer = [{'weights':[np.random.rand() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
    network.append(hidden_layer)
    output_layer = [{'weights':[np.random.rand() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
    network.append(output_layer)

    # hidden_layer = [{'weights':[1 for i in range(n_inputs + 1)]} for i in range(n_hidden)]
    # network.append(hidden_layer)
    # output_layer = [{'weights':[1 for i in range(n_hidden + 1)]} for i in range(n_outputs)]
    # network.append(output_layer)
    return network

def print_network(net):
    for layer in net:
        print(layer)

# Calculate neuron activation for an input
def forward(weights, inputs):
    activation = weights[-1] #the last is bias
    for i in range(len(weights)-1):
        activation += weights[i] * inputs[i]
    return activation

# Forward propagate input to a network output
def forward_propagate(network, row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = forward(neuron['weights'], inputs)
            neuron['output'] = active_fun(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs

def active_fun(x):
    return 1.0 / (1.0 + np.exp(-x)) # sigmoid
    #return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x)) # tanh
    #return np.log(1 + np.exp(x)) # softplus

# Calculate the derivative of an neuron output
def active_derivative(output):
    return output * (1.0 - output)  # x(1-x) is the derivative of sigmod fuc()
    #return 1 - output**2 # tanh
    #return 1.0 / (1.0 + np.exp(-output)) # softplus

# Backpropagate error and store in neurons
def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = []
        if i != len(network)-1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            for j, neuron in enumerate(layer):
                errors.append(neuron['output'] - expected[j])

        for j, neuron in enumerate(layer):
            neuron['delta'] = errors[j] * active_derivative(neuron['output'])

def update_weights(network, row, l_rate):
    """# weight = weight - learning_rate * error * input
       # Update network weights with error
    """
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] -= l_rate * neuron['delta'] * inputs[j]
            neuron['weights'][-1] -= l_rate * neuron['delta']

# Train a network for a fixed number of epochs
def train_network(network, data, n_outputs, l_rate=0.5, n_epoch=10):
    for epoch in range(n_epoch):
        sum_error = 0
        for row in data:
            outputs = forward_propagate(network, row)
            expected = [0 for i in range(n_outputs)]
            expected[row[-1]] = 1
            sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
            backward_propagate_error(network, expected)
            update_weights(network, row, l_rate)
        #print('>epoch=%d, lr=%.5f, loss=%.5f' % (epoch, l_rate, sum_error))
        print(f'>epoch={epoch}, lr={l_rate}, loss={sum_error}')

# Make a prediction with a network
def predict(network, row):
    outputs = forward_propagate(network, row)
    return outputs.index(max(outputs))

def test():
    net = initialize_network(1, 1, 1)
    print_network(net)

    row = [1, 0]
    output = forward_propagate(net, row)
    print('output=', output)

    target = [0, 1]
    backward_propagate_error(net, target)
    print_network(net)

def train_data():
    dataset = [[2.7810836,2.550537003,0],
    [1.465489372,2.362125076,0],
    [3.396561688,4.400293529,0],
    [1.38807019,1.850220317,0],
    [3.06407232,3.005305973,0],
    [7.627531214,2.759262235,1],
    [5.332441248,2.088626775,1],
    [6.922596716,1.77106367,1],
    [8.675418651,-0.242068655,1],
    [7.673756466,3.508563011,1]]

    n_inputs = len(dataset[0]) - 1
    n_outputs = len(set([row[-1] for row in dataset]))
    network = initialize_network(n_inputs, 2, n_outputs)
    train_network(network, dataset, n_outputs, 0.8, 40 )

    for row in dataset:
        prediction = predict(network, row)
        print(f'Expected={row[-1]}, Got={prediction}')

def main():
    #test()
    train_data()

if __name__ == "__main__":
    main()
