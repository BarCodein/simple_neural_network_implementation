import numpy as np
def Relu(x):
    return np.where(x>0,x,0)

def de_Relu(x):
    t =  np.where(x>0,1,0)
    return t

def sigmod(x):
    return 1/(1+ np.exp(-x))

def de_sigmod(x):
    return x * (1 - x)

def liner(x):
    return x

def de_liner(x):
    return x/x
