
#@millarscandrett - basic mnist neural network

from keras.utils.np_utils import to_categorical
import numpy as np
from sklearn.model_selection import train_test_split
import time
from scipy.special import expit
import struct


def relu(inputs):
    return np.maximum(0, inputs)

def d_relu(z, derivative=False):
    return (z > 0).astype(int)

def mnist_loader():
    with open('train-labels-idx1-ubyte', 'rb') as labels:
        magic, n = struct.unpack('>II', labels.read(8))
        train_labels = np.fromfile(labels, dtype=np.uint8)
    with open('train-images-idx3-ubyte', 'rb') as imgs:
        magic, num, nrows, ncols = struct.unpack('>IIII', imgs.read(16))
        train_images = np.fromfile(imgs, dtype=np.uint8).reshape(num,784)
    with open('t10k-labels-idx1-ubyte', 'rb') as labels:
        magic, n = struct.unpack('>II', labels.read(8))
        test_labels = np.fromfile(labels, dtype=np.uint8)
    with open('t10k-images-idx3-ubyte', 'rb') as imgs:
        magic, num, nrows, ncols = struct.unpack('>IIII', imgs.read(16))
        test_images = np.fromfile(imgs, dtype=np.uint8).reshape(num,784)
    return train_images, train_labels, test_images, test_labels

x, y, x_val, y_val = mnist_loader()
x = (x/255).astype('float32')
y = to_categorical(y) #One hot vector encoding
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.15, random_state=42)

class Network(object):

    def __init__(self, n_size, epochs=10, l_rate=0.001):
        self.sizes = n_size
        self.epochs = epochs
        self.l_rate = l_rate

        self.parameters = self.initialisation() #Save all parameters in the NN in this dictionary

    def sigmoid(self, x, derivative=False):
        if derivative:
            return (np.exp(-x))/((np.exp(-x)+1)**2)
        return 1/(1 + np.exp(-x))

    def softmax(self, x, derivative = False):
        exps = np.exp(x - x.max())
        if derivative:
            return exps / np.sum(exps, axis=0) * (1 - exps / np.sum(exps, axis=0))
        return exps / np.sum(exps, axis=0)

    def initialisation(self):
        #number of nodes in our layers

        input_layer = self.sizes[0]
        hidden_1 = self.sizes[1]
        hidden_2 = self.sizes[2]
        output_layer = self.sizes[3]

        #Here we randomly initialise weights based on the dimensions of our weight
        #matrices
        self.bias1 = np.zeros((hidden_1, ))
        self.bias2 = np.zeros((hidden_2,))
        self.bias3 = np.zeros((output_layer,))
        parameters = {
            'w1':np.random.randn(hidden_1, input_layer) * np.sqrt(1. / hidden_1),
            'w2':np.random.randn(hidden_2, hidden_1) * np.sqrt(1. / hidden_2),
            'w3':np.random.randn(output_layer, hidden_2) * np.sqrt(1. / output_layer),
            'b1':self.bias1,
            'b2':self.bias2,
            'b3':self.bias3
        }

        return parameters

    def feedforward(self, x_train):
        #A forward pass mainly consists of using the dot product operation in numpy 
        #We multiply ws by as of prev layer, then apply the activation func to the outcome
        #we use sigmoid squishification as our activation function and softmax 
        #on our last layer to obtain a probability

        parameters = self.parameters

        #input layer activations
        parameters['a0']= x_train
        #input layer -> hidden layer 1 
        parameters['z1'] = np.dot(parameters['w1'], parameters['a0']) + parameters['b1']
        parameters['a1'] = relu(parameters['z1'])
        #hidden layer 1 to hidden layer 2
        parameters['z2'] = np.dot(parameters['w2'], parameters['a1'])  + parameters['b2']
        parameters['a2'] = relu(parameters['z2'])

        #hidden layer 2 to output layer 
        parameters['z3'] = np.dot(parameters['w3'], parameters['a2']) + parameters['b3']
        parameters['a3'] = self.softmax(parameters['z3'])

        return parameters['a3']

    def backpropogation(self, y_train, outpt):
        #outpt is the output from our forward pass
        #backpropogation algorithm allows us to update and adjust hyperparameters + weights
        parameters = self.parameters
        change_w = {}
        #Calculate our w3 update 
        error = 2 * (outpt - y_train) / outpt.shape[0] * self.softmax(parameters['z3'], derivative=True)
        change_w['w3'] = np.outer(error, parameters['a2'])

        #calculate our b1 update
        change_w['b3'] = 1 * error

        #calculate W2 update
        error = np.dot(parameters['w3'].T, error) * d_relu(parameters['z2'], derivative=True)
        change_w['w2'] = np.outer(error, parameters['a1'])

        #calculate our b2 update
        change_w['b2'] = 1 * error

        #calculate w1 update
        error = np.dot(parameters['w2'].T, error) * d_relu(parameters['z1'].T, derivative=True)
        change_w['w1'] = np.outer(error, parameters['a0'])

        #calculate our b3 update
        change_w['b1'] = 1 * error

        return change_w

    def train_net(self, x_train, y_train, x_val, y_val):
        for iteration in range(self.epochs):
            for x,y in zip(x_train, y_train):
                outpt = self.feedforward(x)
                changes_to_w = self.backpropogation(y, outpt)
                self.update_params(changes_to_w)

            accuracy = self.compute_accuracy(x_val,y_val)
            print(f'Epoch:{iteration+1}, accuracy:{accuracy*100}')


    def update_params(self, changes_to_w):
        for key, value in changes_to_w.items():
            if key == 'b1' or 'b2' or 'b3':
                self.parameters[key] = self.parameters[key] - self.l_rate * value
            self.parameters[key] = self.parameters[key] - self.l_rate * value

    def compute_accuracy(self, x_val, y_val):
        predictions = []

        for x, y in zip(x_val, y_val):
            output = self.feedforward(x)
            pred = np.argmax(output)
            predictions.append(pred == np.argmax(y))
        
        return np.mean(predictions)



neural_net = Network(n_size=[784,128,64,10])
neural_net.train_net(x_train, y_train, x_val, y_val)
