
import numpy as np

def trimf(x, a, b, c):
    return np.maximum(np.minimum((x-a)/(b-a+1e-12), (c-x)/(c-b+1e-12)), 0.0)

def gaussmf(x, c, sigma):
    return np.exp(-0.5*((x-c)/(sigma+1e-12))**2)
