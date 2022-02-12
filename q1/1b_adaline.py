import random
import numpy as np
import math
from training_data import data
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


# Let the the perceptron take an input of size 4 with learning rate 0.4.
INPUT_SIZE = 3
LEARNING_RATE = 0.6

def sigmoid(net):
    return 1 / (1 + math.exp(-1*net))

class Perceptron:
    def __init__(self):
        self.weights = np.empty(INPUT_SIZE)
        self.bias = 0
    
    def randomizeWeightsBias(self):
        for weightIndex in range(INPUT_SIZE):
            self.weights[weightIndex] = (random.uniform(-1, 1))        
        self.bias = random.uniform(-1, 1)
    
    def printWeightBias(self):
        print("Weights: ", self.weights)
        print("Bias: ", self.bias, "\n")
    
    def train(self, inputs):
        for i in range(100): # 100 epochs
            for trainingPattern in inputs:
                rawInput = trainingPattern[0]
                if len(rawInput) != INPUT_SIZE:
                    rawInput.pop(0)
                input = np.array(rawInput)
                target = trainingPattern[1]            
                total = 0
                total = np.dot(input, self.weights)
                total -= self.bias
                # sigmoid output
                s = sigmoid(total)
                
                # weight update
                weightUpdateVal = LEARNING_RATE * (float(target) - s)*(pow(s,2)*math.exp(-1*total)) * input
                biasUpdateVal = -1 * LEARNING_RATE * (float(target) - s)*(pow(s,2)*math.exp(-1*total))
                self.weights += weightUpdateVal
                self.bias += biasUpdateVal
            self.printWeightBias()



if __name__ == "__main__":
    neuron = Perceptron()
    neuron.randomizeWeightsBias()
    neuron.train(data)
    # Creating figure
    fig = plt.figure(figsize = (10, 7))
    ax = plt.axes(projection ="3d")

    for dataPoint in data:
        # Creating plot
        point = dataPoint[0]
        if dataPoint[1] == 1:
            ax.scatter3D(point[0], point[1], point[2], color = "green") # class 2
        elif dataPoint[1] == 0:
            ax.scatter3D(point[0], point[1], point[2], color = "red") # class 1
    
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-2.5, 0.5, 100)
    x, y = np.meshgrid(x, y)
    x_coeff = neuron.weights[0]
    y_coeff = neuron.weights[1]
    z_coeff = neuron.weights[2]
    bias = neuron.bias
    equation = (x_coeff/(-1*z_coeff))*x + (y_coeff/(-1*z_coeff))*y - bias/(-1*z_coeff)
    print('z = ', (x_coeff/(-1*z_coeff)), 'x ', (y_coeff/(-1*z_coeff)), 'y +', - bias/(-1*z_coeff))
    ax.plot_surface(x, y, equation)
    plt.title("Scatter plot with adaline boundary")
    ax.set_xlabel('$X$', fontsize=20)
    ax.set_ylabel('$Y$', fontsize=20)
    ax.set_zlabel('$Z$', fontsize=20)
    # new point belonging to class 1
    newPoint = [-1.4,- 1.5, 2]
    ax.scatter3D(newPoint[0], newPoint[1], newPoint[2], color = "blue") # should belong to red class   

    # show plot
    plt.show()

