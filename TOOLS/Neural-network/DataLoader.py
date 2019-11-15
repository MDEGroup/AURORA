# Load CSV
import numpy as np

# load training and testing data from CSV files

def load_data(path, num_of_inputs, num_of_categories):
    """==============read training data=============="""
    raw_data = open(path+'/training_data.csv', 'rt')
    tr_d = np.loadtxt(raw_data, delimiter=",")
    training_inputs = [np.reshape(x, (num_of_inputs, 1)) for x in tr_d]
    raw_data = open(path+'/training_labels.csv', 'rt')
    tr_l = np.loadtxt(raw_data, delimiter=",")
    training_labels = [one_hot_vector(y,num_of_categories) for y in tr_l]
    training_data = zip(training_inputs, training_labels)
    """==============read testing data=============="""
    raw_data = open(path+'/testing_data.csv', 'rt')
    te_d = np.loadtxt(raw_data, delimiter=",")
    testing_inputs = [np.reshape(x, (num_of_inputs, 1)) for x in te_d]

    raw_data = open(path+'/testing_labels.csv', 'rt')
    te_l = np.loadtxt(raw_data, delimiter=",")
    testing_data = zip(testing_inputs, te_l)

    return (training_data, testing_data)

def one_hot_vector(j,num_of_categories):    
    e = np.zeros((num_of_categories, 1))
    e[int(j)] = 1.0
    return e
