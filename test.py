import numpy as np
delta = np.array([1, 2, 3, 4])
activation = np.array([1, 2, 3, 4])
print(np.dot(delta, activation.transpose()))