import DataLoader
import NeuralNetwork as network
import datetime

def run(path, num_of_inputs, num_of_2nd_layer_neurons, num_of_categories, num_of_epochs, learning_rate, expected_accuracy):	
	training_data, test_data = DataLoader.load_data(path, num_of_inputs, num_of_categories)
	time1 = datetime.datetime.now()
	net = network.Network(path,expected_accuracy,[num_of_inputs, num_of_2nd_layer_neurons, num_of_categories])
	net.SGD(training_data, num_of_epochs, learning_rate, test_data=test_data)
	time2 = datetime.datetime.now()
	elapsedTime = time2 - time1	
	print "The total execution time is: {0}".format(elapsedTime.total_seconds())


def rerun(path, num_of_inputs, num_of_2nd_layer_neurons, num_of_categories, num_of_epochs, learning_rate, expected_accuracy):
	training_data, test_data = DataLoader.load_data(path, num_of_inputs, num_of_categories)
	net = network.Network(path,expected_accuracy,[num_of_inputs, num_of_2nd_layer_neurons, num_of_categories])
	net.load(path)
	results = net.SGD(training_data, num_of_epochs, learning_rate, test_data=test_data)	
	net.saveResults(path, results)