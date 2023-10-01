import numpy as np
from random import randrange, shuffle
from math import ceil, floor
from copy import deepcopy

#Sigmoid activation function
def sigmoid(z, d=False):
    if d:
        return(sigmoid(z) * (1.0 - sigmoid(z)))
    z = np.clip(z, -500, 500)
    return(1.0 / (1.0 + np.exp(-z)))

def linear(z, d=False):
    if d:
        return(1)
    return(z)

#Rectified Linear Unit (ReLU) activation function
def relu(z, d=False):
    bool_array = z > 0
    if d:
        return((bool_array * 0.99) + 0.01)
    return(((bool_array * 0.99) + 0.01) * z)

#Hyperbolic tangent (tanh) activation function
def tanh(z, d=False):
    if d:
        return(1 - tanh(z)**2)
    z = np.clip(z, -500, 500)
    return((np.exp(z) - np.exp(-z)) / ((np.exp(z)+np.exp(-z))))

#soft max final layer activation function
def soft_max(z, d=False):
    if d:
        return(1)
    exps = np.exp(z - np.max(z, axis=0))
    return(exps / np.sum(exps, axis=0))

#Cross entropy cost function (only the derivative with respect to soft max)
def cross_entropy(output, expected, d=False):
    if d:
        return(output - expected)

#Mean Squared Error cost function
def cost(output, expected, d=False):
    if d:
        return(output - expected)
    return((output - expected) ** 2)

def crossover(nn1, nn2, crr):
    #crr is crossover rate
    for w1, w2 in zip(nn1.weights, nn2.weights):
        for j in range(np.size(w1, 0)):
            for k in range(np.size(w1, 1)):
                if np.random.rand() < crr:
                    a = w1[j][k]
                    b = w2[j][k]
                    w1[j][k] = b
                    w2[j][k] = a
    for b1, b2 in zip(nn1.biases, nn2.biases):
        for i in range(np.size(nn1.biases, 0)):
            if np.random.rand() < crr:
                a = b1[i][0]
                b = b2[i][0]
                b1[i][0] = b
                b2[i][0] = a


class Network:

    def __init__(self, list_of_nodes, activ_func=sigmoid, ol_activ_func=sigmoid):
        #LON is a list with each index referring to how many nodes are in that layer.
        #For example, [2, 5, 3] will make an MLP with 2 input nodes, 1 hidden layer with 5 nodes, and 3 output nodes
        if type(list_of_nodes) == Network:
            self.lon = list_of_nodes.lon
            self.num_layers = len(self.lon)
            self.activ_func = list_of_nodes.activ_func
            self.ol_activ_func = list_of_nodes.ol_activ_func
            self.weights = deepcopy(list_of_nodes.weights)
            self.biases = deepcopy(list_of_nodes.biases)
        else:
            self.lon = list_of_nodes
            self.activ_func = activ_func
            self.ol_activ_func = ol_activ_func
            #Initializing weights and biases with empty lists
            self.weights = []
            self.biases = []
            self.num_layers = len(self.lon)
            #Initializing random weight and bias matrices for all layers
            for i in range(1, self.num_layers):
                self.weights.append(np.random.normal(0.0, pow(self.lon[i - 1], -0.5), (self.lon[i], self.lon[i - 1])))
                self.biases.append(np.full((self.lon[i], 1), 0.1))
        #Initializing  zs and activations with empty lists
        self.zs = []
        self.activations = []
        self.mu, self.sigma = 0, 1

    #This function feeds the inputs into the MLP to get the outputs using some matrix math
    def feedforward(self, input):
    	#Input matrix is a vector with length lon[0]
        activation = input
        self.activations.append(activation)
        #For loop passes inputs through all but the last layer (because last layer may have special activation function)
        for w, b in zip(self.weights[0:-1], self.biases[0:-1]):
            z = np.dot(w, activation) + b
            self.zs.append(z)
            activation = self.activ_func(z)
            self.activations.append(activation)
        #Passing activation values through last layer with the last layer's activation function (ol_activ_func)
        w, b = self.weights[-1], self.biases[-1]
        z = np.dot(w, activation) + b
        self.zs.append(z)
        activation = self.ol_activ_func(z)
        self.activations.append(activation)
        return

    #This function backpropagates through MLP to tweak and correct weights/biases
    def backprop(self, output):
        #Initializing change_bs and change_ws which are going to be the changes to biases and weights
        change_ws = [np.zeros(w.shape) for w in self.weights]
        change_bs = [np.zeros(b.shape) for b in self.biases]
        #Derivative of last layer z values with respect to cost function (derivative of cost function times derivative of activation function)
        change_z = cost(self.activations[-1], output, d=True) * self.ol_activ_func(self.zs[-1], d=True)
        #Derivative of last layer biases with respect to cost (happens to be same as change_z)
        change_bs[-1] = change_z
        #Derivative of last layer weights with respect to cost (change_z times activations of previous layer)
        change_ws[-1] = np.dot(change_z, self.activations[-2].T)
        #Deriving change_z, change_ws, and change_bs for all other layers
        for i in range(2, self.num_layers):
            dz = self.activ_func(self.zs[-i], d=True)
            change_z = np.dot(self.weights[-i + 1].T, change_z) * dz
            change_bs[-i] = change_z
            change_ws[-i] = np.dot(change_z, self.activations[-i - 1].T)
        #Adjusting weights by subtracting change_ws(multiplied by the learning rate) from weights
        self.weights = [w - change_w * self.lr for w, change_w in zip(self.weights, change_ws)]
        #Adjusting biases by subtracting change_bs(multiplied by the learning rate) from biases
        self.biases = [b - np.sum(change_b, axis=1, keepdims=True) * self.lr for b, change_b in zip(self.biases, change_bs)]
        return

    def matrixize(self, orig_xy_set):
        xy_set = [(np.array([x]).T, np.array([y]).T) for x, y in orig_xy_set]
        return(xy_set)

    def normalize(self, xy_set):
        inputs = [set[0] for set in xy_set]
        inputs = np.concatenate(inputs, axis=1)
        self.mu = np.mean(inputs)
        self.sigma = np.std(inputs)
        if self.sigma == 0.0:
            self.sigma = 1.0
        return

    def train(self, train_data, epochs, learning_rate, minibatch_size=1, test_data=None, normalize=False, collect_wrong=False):
        self.lr = learning_rate
        train_xy_set = self.matrixize(train_data)
        #Normalize inputs if needed (if normalize=True)
        if normalize:
            self.normalize(train_xy_set)
        self.epochs = epochs
        self.minibatch_size = minibatch_size
        for i in range(epochs):
            shuffle(train_xy_set)
            for k in range(floor(len(train_xy_set) / self.minibatch_size)):
                batch = train_xy_set[k * self.minibatch_size:(k + 1) * self.minibatch_size]
                train_input = np.concatenate([set[0] for set in batch], axis=1)
                train_input = (train_input - self.mu) / self.sigma
                train_output = np.concatenate([set[1] for set in batch], axis=1)
                self.feedforward(train_input)
                self.backprop(train_output)
        if test_data:
            test_xy_set = self.matrixize(test_data)
            test_input = np.concatenate([set[0] for set in test_xy_set], axis=1)
            test_input = (test_input - self.mu) / self.sigma
            test_output = np.concatenate([set[1] for set in test_xy_set], axis=1)
            self.test((test_input, test_output), collect_wrong)
        return

    def query(self, input):
        input = np.array([input]).T
        input = (input - self.mu) / self.sigma
        self.feedforward(input)
        return(self.activations[-1].T[0].tolist())

    def get_accuracy(self, xy_set, collect_wrong):
        input, output = xy_set
        self.feedforward(input)
        predicted_label = np.argmax(self.activations[-1], axis=0)
        expected_label = np.argmax(output, axis=0)
        score = np.sum(predicted_label == expected_label)
        if collect_wrong:
            input = np.hsplit(input, input.shape[1])
            self.wrong_inputs = np.where(predicted_label != expected_label)
            self.wrong_inputs = [input[i] for i in self.wrong_inputs[0]]
            self.wrong_inputs = [value * self.sigma + self.mu for value in self.wrong_inputs]
            self.wrong_inputs = [value.T[0].tolist() for value in self.wrong_inputs]
        return(score)

    def test(self, xy_set, collect_wrong):
        accuracy = self.get_accuracy(xy_set, collect_wrong)
        print('Topology: {0}'.format(self.lon))
        print('Epochs: {0}'.format(self.epochs))
        print('Learning Rate: {0}'.format(self.lr))
        print('Accuracy: {0} / {1} = {2}%'.format(accuracy, xy_set[0].shape[1], (accuracy / xy_set[0].shape[1]) * 100))

    def mutate(self, mutation_rate):
        for weights in self.weights:
            for j in range(np.size(weights, 0)):
                for k in range(np.size(weights, 1)):
                    if np.random.rand() < mutation_rate:
                        weights[j][k] += np.random.normal(0, 0.1)
        return(self)
