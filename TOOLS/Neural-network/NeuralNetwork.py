"""
This modul has been built mainly based on the following source code

network.py

By Michael Nielsen
https://github.com/mnielsen/neural-networks-and-deep-learning.git

and
Python Machine Learning Tutorial
https://www.python-course.eu/neural_network_mnist.php

"""

import numpy as np
import pandas as pd
import random


# compute sigmoid function
def sigmoid(input):    
    output = 1/(1+np.exp(-input))
    return output   

# compute derivate
def sigmoid_derivative(input):    
    return sigmoid(input)*(1-sigmoid(input))


class Network(object):    
    #===============initialize the network's parameters===============
    # root: the directory that contains the source code
    # e_accuracy: the expected classification accuracy you want to obtain, e.g., 0.9 
    # num_layers: the number of layers for the network
    # sizes: the size of each layer
    # biases
    # weights
    def __init__(self, path, expected_accuracy, sizes):
        self.root = path
        self.e_accuracy = expected_accuracy
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    # compute the corresponding output, given an input a
    def feedforward(self, a):        
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a


    def SGD(self, training_data, epochs, mini_batch_size, learning_rate, test_data=None):
        
        if test_data: num_of_testing_items = len(test_data)
        n = len(training_data)

        max = 0

        #training phase
        e = 0

        # keep running while the maximum accuracy is smaller than a pre-defined value, 
        # or the num of epochs is not reached

        while max < self.e_accuracy or e < epochs: 
            print "epoch {0}".format(e)
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.refine(mini_batch, learning_rate)
            
            #get the classification results
            classification_results = self.predict(test_data)

            #number of true positives
            num_of_TP = sum(int(x == y) for (x, y) in classification_results)

            #success rate
            success_rate = float(num_of_TP)/num_of_testing_items

            if success_rate > max :
                max = success_rate
                print "Found a better success rate {0}".format(success_rate)
                #save the weights, parameters to external files if a better accuracy has been reached
                self.save(self.root)
            e = e+1

        print "The best success rate is: {0}".format(max)

        return classification_results
               
      
    # refine the network's weights and biases
    def refine(self, mini_batch, learning_rate):
        
        synapse_b = [np.zeros(b.shape) for b in self.biases]
        synapse_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            error_synapse_b, error_synapse_w = self.back_propagation(x, y)
            synapse_b = [nb+dnb for nb, dnb in zip(synapse_b, error_synapse_b)]
            synapse_w = [nw+dnw for nw, dnw in zip(synapse_w, error_synapse_w)]
        
        self.weights = [w-(learning_rate/len(mini_batch))*nw
                        for w, nw in zip(self.weights, synapse_w)]
        self.biases = [b-(learning_rate/len(mini_batch))*nb

                       for b, nb in zip(self.biases, synapse_b)]


    def back_propagation(self, x, y):
        
        synapse_b = [np.zeros(b.shape) for b in self.biases]
        synapse_w = [np.zeros(w.shape) for w in self.weights]
        
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
        error = self.error(activations[-1], y) * \
            sigmoid_derivative(zs[-1])
        synapse_b[-1] = error
        synapse_w[-1] = np.dot(error, activations[-2].transpose())

        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_derivative(z)
            error = np.dot(self.weights[-l+1].transpose(), error) * sp
            synapse_b[-l] = error
            synapse_w[-l] = np.dot(error, activations[-l-1].transpose())
        return (synapse_b, synapse_w)

    
    # predict labels for testing data
    def predict(self, test_data):
        results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        
	"""for r in test_results:
              if(r[0]==r[1]):
 		  print(r) 
              else:
		  print(r)"""
        #return sum(int(x == y) for (x, y) in test_results)
        return results
     

    # error between the predicted labels and the real labels
    def error(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

    
    # save all weights and biases to files 
    def save(self,path):
        np.save(path+'/saved_num_layers.npy', self.num_layers)
        np.save(path+'/saved_sizes.npy', self.sizes)
        np.save(path+'/saved_weights.npy', self.weights)
        np.save(path+'/saved_biases.npy', self.biases)
        pass
    
    # load neural network weights and biases from files
    def load(self,path):
        self.num_layers = np.load(path+'/saved_num_layers.npy')
        self.sizes = np.load(path+'/saved_sizes.npy')
        self.weights = np.load(path+'/saved_weights.npy')
        self.biases = np.load(path+'/saved_biases.npy')
        pass


    # save classification results to an external file
    def saveResults(self, path, results):
        mat = np.matrix(results)
        dataframe = pd.DataFrame(data=mat.astype(float))
        dataframe.to_csv(path+'/Results.csv', sep=' ', header=False, float_format='%.2f', index=False) 


