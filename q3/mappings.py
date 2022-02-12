from cmath import sin
import matplotlib.pyplot as plt
import numpy as np


import math

def f1(x):
    return x * math.sin(6*math.pi*x) * math.exp(-1*pow(x,2))

def f2(x):
    return math.exp(-1*pow(x,2)) * math.atan(x) * math.sin(4*math.pi*x)
