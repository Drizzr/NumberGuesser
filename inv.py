from training.data import data
from mis_loader import vectorized_result
from network import Network
import random
import time
import numpy as np

random.shuffle(data)


trainig_data = data[:500]
test_data = data[:-500]


trainig_results = [vectorized_result(y[1], 2) for y in trainig_data]
trainig_inputs = [y[0] for y in trainig_data]

training_data = list(zip(trainig_inputs, trainig_results))

net = Network([9, 100, 20, 2])
net.SGD(training_data, 50, 20, 2, test_data=test_data)

trans = ["not invertible", "invertible"]
random.shuffle(test_data)
for entry in test_data:
    print(f" ({entry[0][:3]}) \n ({entry[0][3:6]})\n ({entry[0][6:9]})")
    int = np.argmax(net.feedforward(entry[0]))
    guess = trans[int]

    print(f"Network guess: {guess}, {int == entry[1]}")

    time.sleep(10)
