import numpy as np
def Relu(x):
    return np.where(x>0,x,0)

def de_Relu(error,value):
    t =  np.where(value>0,1,0)
    return t*error

