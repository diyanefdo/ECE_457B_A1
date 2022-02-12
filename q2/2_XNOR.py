import matplotlib.pyplot as plt
import numpy as np

adaline1Params = [(1, -1), -1.5] # [(weight 1, weight 2), bias]
adaline2Params = [(-1, 1), -1.5]
adalineXORParams = [(-1, -1), -1.5]

stepFunc = lambda num: 1 if num>0 else -1 if num<0 else 0 

def adaline(inputs, weights, bias):
    net = inputs[0]*weights[0] + inputs[1]*weights[1] + bias
    return stepFunc(net)

def madelineXNOR(inputs):
    # two adalines in first layer
    out1 = adaline(inputs, adaline1Params[0], adaline1Params[1])
    out2 = adaline(inputs, adaline2Params[0], adaline2Params[1])
    
    # last layer XOR
    finalOut = adaline((out1, out2), adalineXORParams[0], adalineXORParams[1])
    return finalOut

if __name__ == "__main__":
    inputs = [
        (-1, -1), 
        (-1, 1), 
        (1, -1), 
        (1, 1)
        ]
        
    for input in inputs:
        x1 = input[0]
        x2 = input[1]
        output = madelineXNOR((x1, x2))
        # plot for each output
        if output == 1:
            plt.scatter(x1, x2, marker='o', c='black')
        elif output == -1:
            plt.scatter(x1, x2, marker='x', c='black')

    x1 = np.linspace(-2, 2, 100)
    adaline1_X2 = (adaline1Params[0][0]/(-1*adaline1Params[0][1]))*x1 + adaline1Params[1]/(-1*adaline1Params[0][1])
    adaline2_X2 = (adaline2Params[0][0]/(-1*adaline2Params[0][1]))*x1 + adaline2Params[1]/(-1*adaline2Params[0][1])

    plt.plot(x1, adaline1_X2)
    plt.plot(x1, adaline2_X2)    

    plt.xlabel("X1")
    plt.ylabel("X2")   
    plt.title("XNOR gate results and boundary lines") 
    plt.show()

