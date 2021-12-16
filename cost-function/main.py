import numpy as np
import random
import os
import time

import json
import matplotlib.pyplot as plot;

class NeuralNetwork:

    def hyper_params(self, topology = [784, 100, 10], learning_rate = 3.0, batch_size = 10, epochs = 30, evaluate = True):
        self.topology = topology
        self.layer_amount = len(topology)
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs

        self.evaluate = evaluate
        self.eval_training_cost = 0
        self.eval_start_time = 0
        self.eval_current_time = 0

        self.weights = []
        self.biases = []

        for i in range(len(topology)-1):
            self.weights.append(np.random.randn(topology[i+1], topology[i]))
            self.biases.append(np.random.randn(topology[i+1], 1))

    def start_training(self, training_data, validation_data = []):

        print('Starting Network Training...')
        training_data = list(training_data)

        training_cost_x_values = []
        training_cost_y_values = []

        training_classification_x_values = []
        training_classification_y_values = []

        validation_classification_x_values = []
        validation_classification_y_values = [] 

        validation_cost_x_values = []
        validation_cost_y_values = []

        self.eval_start_time = time.perf_counter()

        for epoch in range(1, self.epochs+1):

            if (self.evaluate and epoch == 2): # Beräknad tid att genomföra programmet
                print(f"Estimated seconds left: {(time.perf_counter() - self.eval_start_time)*(self.epochs-4):0.1f}")

            random.shuffle(training_data)

            batches = []
            for i in range(0, len(training_data), self.batch_size):
                batches.append([])
                for y in range(i, i+self.batch_size):
                    batches[-1].append(training_data[y])
            i = 1
            for batch in batches:
                self.propagate_network(batch)
                i += 1
            
            print('Epoch', epoch, 'is complete')

            if (self.evaluate):
                if validation_data:
                    validation_accuracy, validation_cost = self.test_network_internal(validation_data)

                    validation_classification_x_values.append(epoch)
                    validation_classification_y_values.append(validation_accuracy)

                    validation_cost_x_values.append(epoch)
                    validation_cost_y_values.append(validation_cost)
                
                training_accuracy, training_cost = self.test_network_internal(training_data)

                training_cost_x_values.append(epoch)
                training_cost_y_values.append(training_cost)

                training_classification_x_values.append(epoch)
                training_classification_y_values.append(training_accuracy)
        
        if (self.evaluate):

            self.eval_current_time = time.perf_counter()

            dirPath = "results/" + str(self.topology) + " " + str(self.learning_rate) + " " + str(self.batch_size) + " " + str(self.epochs)
            os.mkdir(dirPath)

            self.save_params(dirPath + '/params.json')

            with open(dirPath + '/data.txt', 'a') as file:
                file.write('Start training cost: ' + str(training_cost_y_values[0]) + '\n')
                file.write('Final training cost: ' + str(training_cost_y_values[-1]) + '\n')
                file.write('Training classification accuracy: ' + str(training_classification_y_values[-1]))
                file.write('\n\n')
                file.write('Start validation cost: ' + str(validation_cost_y_values[0]) + '\n')
                file.write('Final validation cost: ' + str(validation_cost_y_values[-1]) + '\n')
                file.write('Validation classification accuracy: ' + str(validation_classification_y_values[-1]))
                file.write('\n\n')
                file.write(f'Time elapsed (seconds): {self.eval_current_time - self.eval_start_time:0.4f}')

            plot.figure()
            plot.xlabel('Epok')
            plot.ylabel('Kostnad')

            plot.plot(training_cost_x_values, training_cost_y_values)
            plot.savefig(dirPath + '/training-kostnad.jpg')

            plot.figure()
            plot.ylabel('Klassifikation (%)')
            plot.xlabel('Epok')

            plot.plot(training_classification_x_values, training_classification_y_values)
            plot.savefig(dirPath + '/training-klassifikation.png')

            if validation_data:
                plot.figure()

                plot.ylabel('Klassifikation (%)')
                plot.xlabel('Epok')

                plot.plot(validation_classification_x_values, validation_classification_y_values)
                plot.savefig(dirPath + '/validation-klassifikation.png')

                plot.figure()

                plot.xlabel('Epok')
                plot.ylabel('Kostnad')

                plot.plot(validation_cost_x_values, validation_cost_y_values)
                plot.savefig(dirPath + '/validation-kostnad.jpg')
    
    def propagate_network(self, batch):
        gradient_weights = []
        gradient_biases = []

        for i in range(len(self.weights)):
            gradient_weights.append(np.zeros(self.weights[i].shape))
        for i in range(len(self.biases)):
            gradient_biases.append(np.zeros(self.biases[i].shape))

        cost = 0

        for sample in batch:
            delta_gradient_weights, delta_gradient_biases = self.back_propagation(sample)

            for i in range(len(gradient_weights)):
                gradient_weights[i] += delta_gradient_weights[i]
            for i in range(len(gradient_biases)):
                gradient_biases[i] += delta_gradient_biases[i]
            
        # Update the weights and biases
        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i] - (self.learning_rate / self.batch_size) * gradient_weights[i]
        for i in range(len(self.biases)):
            self.biases[i] = self.biases[i] - (self.learning_rate / self.batch_size) * gradient_biases[i]
    
    def back_propagation(self, sample):
        gradient_weights = []
        gradient_biases = []

        activations = []
        activation = sample[0]

        weighted_inputs = []

        for i in range(len(self.weights)):
            gradient_weights.append(np.zeros(self.weights[i].shape))
        for i in range(len(self.biases)):
            gradient_biases.append(np.zeros(self.biases[i].shape))

        activations.append(activation)

        # Propagate through the network
        for i in range(len(self.weights)):
            w_input = np.dot(self.weights[i], activation) + self.biases[i]
            activation = self.sigmoid(w_input)

            activations.append(activation)
            weighted_inputs.append(w_input)

        # Calculate error for output layer
        error = activations[-1] - sample[1] # Nya backprop nr 1 för cross-entropy.
        #error = self.cost_deriv(activations[-1], sample[1]) # Cross-entropy

        # Calculate gradients for output layer
        gradient_weights[-1] = np.dot(error, activations[-2].transpose())
        gradient_biases[-1] = error


        # Loop through layers backwards 
        for i in range(-2, -self.layer_amount, -1):
            w_input = weighted_inputs[i]

            # Calculate error for current layer
            error = np.dot(self.weights[i+1].transpose(), error) * self.sigmoid_deriv(w_input)

            # Calculate gradients for current layer
            gradient_weights[i] = np.dot(error, activations[i-1].transpose())
            gradient_biases[i] = error


        return (gradient_weights, gradient_biases)


    def forward_propagation(self, input):

        activation = input
        for i in range(len(self.weights)):
            w_input = np.dot(self.weights[i], activation) + self.biases[i]
            activation = self.sigmoid(w_input)

        return activation

    def test_network_internal(self, data):
        amount_correct = 0
        cost = 0
        for sample in data:
            image_data = sample[0]
            label_data = sample[1]

            activation = self.forward_propagation(image_data)
            if (np.argmax(activation) == np.argmax(label_data)): amount_correct += 1

            cost += self.cost_cross(activation, label_data)
        
        return (amount_correct/len(data)*100, cost/len(data))

    def test_network(self, test_data):
        accuracy, cost = self.test_network_internal(test_data)
        print("Classification Accuracy:", accuracy, "%")

    def save_params(self, file):
        params = {
                "topology": self.topology,
                "learning_rate": self.learning_rate,
                "batch_size": self.batch_size,
                "epochs": self.epochs,
                "weights": [w.tolist() for w in self.weights],
                "biases" : [b.tolist() for b in self.biases]
        }

        file = open(file, 'w')
        json.dump(params, file, indent=4)
        file.close()
    
    def sigmoid(self, input):
        return 1.0 / (1.0 + np.exp(-input))
    
    def sigmoid_deriv(self, input):
        return self.sigmoid(input) * (1 - self.sigmoid(input))
    
    def cost_cross(self, activation, desired_activation):
        return -np.sum(np.nan_to_num(desired_activation * np.log(activation) + (1 - desired_activation)*np.log(1 - activation))) 


def load_network_instance():

    file = open('params.json', 'r')
    params = json.load(file)
    file.close()

    network = NeuralNetwork()
    network.hyper_params(params["topology"], params["learning_rate"], params["batch_size"], params["epochs"])
    
    network.weights = [np.array(w) for w in params["weights"]]
    network.biases = [np.array(b) for b in params["biases"]]

    return network

import import_mnist

train = False


if (train):
    net = NeuralNetwork()
    training_data, validation_data = import_mnist.import_training_samples(60000)
    net.hyper_params([784, 100, 30, 10], 0.65, 10, 100) # topologi, learning-rate, batch-size, epoker
    net.start_training(training_data, validation_data)
    net.save_params('params.json')
else:
    net = load_network_instance()

    test_data = import_mnist.import_test_samples(10_000)
    net.test_network(test_data)