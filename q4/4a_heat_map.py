import numpy as np
import matplotlib.pyplot as plt

test_accuracies = np.array([[0.8888889,  0.95555556, 0.95555556, 0.97777778],
 [0.53333336, 0.93333334, 0.97777778, 0.93333334],
 [0.68888891, 0.93333334, 0.93333334, 0.95555556],
 [0.68888891, 0.53333336, 0.97777778, 0.97777778]])

plt.imshow(test_accuracies, cmap='hot', interpolation='nearest')
plt.show()