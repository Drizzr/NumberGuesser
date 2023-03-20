import numpy as np
import random

from mis_loader import load_data_wrapper

class Network:

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(1, y) for y in sizes[1:]]
        self.weights = [np.random.randn(x, y) 
                        for x, y in zip(sizes[:-1], sizes[1:])]


    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):

            a = sigmoid(np.dot(w.T, a).T+b[0])
        return a
    
    def SGD(self, training_data, controll_data, epochs, mini_batch_size, eta, test_data=None):
        if test_data: n_test = len(training_data)
        n = len(training_data)
        mini_batches = []
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
                #for k in range(0, n, mini_batch_size))
            controll = [controll_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            data = zip(mini_batches, controll)
            for mini_batch, controll in data:
                self.update_mini_batch(np.array(mini_batch), np.array(controll), eta)
            if test_data:
                print(f"Epoch {j}: {self.evaluate(test_data)} / {n_test}")
            else:
                print(f"Epoch {j} complete")

    def update_mini_batch(self, mini_batch, controll, eta):

        delta_nabla_b, delta_nabla_w = self.backprop(mini_batch, controll)

        self.weights = [w-eta*nw
                        for w, nw in zip(self.weights, delta_nabla_w)]
        self.biases  = [b-eta*nb
                        for b, nb in zip(self.biases, delta_nabla_b)]
    
    
    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)
    

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] 
        zs = [] 
        for b, w in list(zip(self.biases, self.weights)):
            z = np.dot(activations[-1], w) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        

        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta.mean(0)
        #print((self.biases[-1]-nabla_b[-1]).shape)
        nabla_w[-1] = np.dot(delta.T, activations[-2]).T / len(y)

        for l in range(2, self.num_layers):
            #print(1)
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1], delta.T).T * sp
            nabla_b[-l] = delta.mean(0)
            #print((self.biases[-l]-nabla_b[-l]).shape)
            nabla_w[-l] = np.dot(delta.T, activations[-l-1]).T / len(y)
        return (nabla_b, nabla_w)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)


def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))
#print(net.weights)

training_data, controll, validation_data, test_data = load_data_wrapper()
net = Network([784, 100, 10])
net.SGD(training_data, controll, 30, 10, 1, test_data=test_data)
