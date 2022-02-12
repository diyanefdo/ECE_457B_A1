import numpy as np
import matplotlib.pyplot as plt

f1_val_errors = np.array([[0.08709891, 0.08558181, 0.10152956, 0.09943398],
 [0.04697868, 0.04824107, 0.05069744, 0.05347855],
 [0.0716501, 0.07330044, 0.07347904, 0.07492599],
 [0.05498801, 0.04810503, 0.04688598, 0.04599991]])

f2_val_errors = np.array([[ 0.02136804,  0.15908674,  0.03, 0.03],
 [ 0.02557761,  0.02643157,  0.02583188,  0.03168853],
 [ 0.02666681,  0.02787745,  0.03036656,  0.02694039],
 [ 0.03148259,  0.028654,    0.03104081,  0.03658469]])

result1 = np.where(f1_val_errors == np.amin(f1_val_errors))
result2 = np.where(f2_val_errors == np.amin(f2_val_errors))

print("f1_min_index: ", result1)
print("f2_min_index: ", result2)

plt.imshow(f1_val_errors, cmap='hot', interpolation='nearest')
plt.show()

plt.imshow(f2_val_errors, cmap='hot', interpolation='nearest')
plt.show()