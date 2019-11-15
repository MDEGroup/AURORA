"""
This modul has been built mainly based on the following source code

1. network.py
https://github.com/mnielsen/neural-networks-and-deep-learning.git

MIT License

Copyright (c) 2012-2018 Michael Nielsen

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, 
including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software
is furnished to do so, subject to the following conditions:

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. 
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


and

2. Python Machine Learning Tutorial
https://www.python-course.eu/neural_network_mnist.php

"""

import numpy as np
import pandas as pd
import random

MINIBATCH_SIZE = 10

# compute the sigmoid function
def sigmoid(input):    
    output = 1/(1+np.exp(-input))
    return output   

# compute the derivate of the sigmoid function
def sigmoid_derivative(input):    
    return sigmoid(input)*(1-sigmoid(input))


class Network(object):    
    #===============initialize the network's parameters===============
    # root: the directory that contains the source code
    # e_accuracy: the expected classification accuracy you want to obtain, e.g., 0.9 
    # num_layers: the number of layers for the network
    # layers: the size of each layer
    # biases
    # weights
    def __init__(self, path, expected_accuracy, layers):
        self.root = path
        self.e_accuracy = expected_accuracy
        self.num_layers = len(layers)
        self.layers = layers
        self.biases = [np.random.randn(y, 1) for y in layers[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(layers[:-1], layers[1:])]

    # compute the corresponding output, given an input a
    def feedforward(self, a):        
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a


    def SGD(self, training_data, epochs, learning_rate, test_data=None):
        
        if test_data: num_of_testing_items = len(test_data)

        num_of_training_items = len(training_data)

        max = 0

        #training phase
        e = 0

        # keep running while the maximum accuracy is smaller than a pre-defined value, 
        # or the num of epochs is not reached

        while max < self.e_accuracy or e < epochs: 
            print "epoch {0}".format(e)
            random.shuffle(training_data)
            
            mini_batches = [
                training_data[k:k+MINIBATCH_SIZE]
                for k in xrange(0, num_of_training_items, MINIBATCH_SIZE)]
            for mini_batch in mini_batches:
                self.refine(mini_batch, learning_rate)
            
            #get the classification results
            classification_results = self.predict(test_data)

            #number of true positives
            num_of_true_positives = sum(int(x == y) for (x, y) in classification_results)

            #accuracy: the ratio of number of correctly classified items to the total number of items in the test set
            accuracy = float(num_of_true_positives)/num_of_testing_items

            if accuracy > max :
                max = accuracy
                print "Found a better accuracy {0}".format(accuracy)
                #save the weights, parameters to external files if a better accuracy has been reached
                self.save(self.root)
            e = e+1

        print "The best accuracy is: {0}".format(max)

        return classification_results
               
      
    # refine the network's weights and biases
    def refine(self, mini_batch, learning_rate):
        
        synapse_b = [np.zeros(b.shape) for b in self.biases]
        synapse_W = [np.zeros(W.shape) for W in self.weights]
        
        for x, y in mini_batch:
            error_synapse_b, error_synapse_W = self.back_propagation(x, y)
            synapse_b = [nb+dnb for nb, dnb in zip(synapse_b, error_synapse_b)]
            synapse_W = [nW+dnW for nW, dnW in zip(synapse_W, error_synapse_W)]
        
        self.weights = [W-(learning_rate/len(mini_batch))*nW
                        for W, nW in zip(self.weights, synapse_W)]
        
        self.biases = [b-(learning_rate/len(mini_batch))*nb

                       for b, nb in zip(self.biases, synapse_b)]


    def back_propagation(self, x, y):
        
        synapse_b = [np.zeros(b.shape) for b in self.biases]
        synapse_W = [np.zeros(W.shape) for W in self.weights]
        
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        
        # backward pass
        error = self.calculate_error(activations[-1], y) * \
            sigmoid_derivative(zs[-1])
        synapse_b[-1] = error

        synapse_W[-1] = np.dot(error, activations[-2].transpose())

        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_derivative(z)
            error = np.dot(self.weights[-l+1].transpose(), error) * sp
            synapse_b[-l] = error
            synapse_W[-l] = np.dot(error, activations[-l-1].transpose())
        return (synapse_b, synapse_W)

    
    # predict labels for testing data
    def predict(self, test_data):
        results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        
	"""for r in test_results:
              if(r[0]==r[1]):
 		  print(r) 
              else:
		  print(r)"""
        #return sum(int(x == y) for (x, y) in test_results)
        return results
     

    # error between the predicted labels and the real labels
    def calculate_error(self, y_pred, y_):
        return (y_pred-y_)

    
    # save all weights and biases to external files 
    def save(self,path):
        np.save(path+'/saved_num_layers.npy', self.num_layers)
        np.save(path+'/saved_layers.npy', self.layers)
        np.save(path+'/saved_weights.npy', self.weights)
        np.save(path+'/saved_biases.npy', self.biases)
        pass
    
    # load neural network weights and biases from external files
    def load(self,path):
        self.num_layers = np.load(path+'/saved_num_layers.npy')
        self.layers = np.load(path+'/saved_layers.npy')
        self.weights = np.load(path+'/saved_weights.npy')
        self.biases = np.load(path+'/saved_biases.npy')
        pass


    # save classification results to an external file, i.e., Results.csv
    def saveResults(self, path, results):
        mat = np.matrix(results)
        dataframe = pd.DataFrame(data=mat.astype(float))
        dataframe.to_csv(path+'/Results.csv', sep=' ', header=False, float_format='%.2f', index=False) 


