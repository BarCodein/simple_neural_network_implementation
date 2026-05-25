import numpy as np
import pandas as pd

class layer:
    def __init__(self,num_output):
        self.output = np.zeros((num_output,1)) #output matrix
        self.num_output = num_output
        self.next_layer = None
    

class input_layer(layer):
    def get_input(self,input_data):
        assert(isinstance(input_data,np.ndarray))
        assert(np.shape(input_data)==np.shape(self.output))
        self.output = input_data + 0

class hidden_layer(layer):
    def __init__(self, num_output,last_layer,fuc,back_fuc):
        super().__init__(num_output)
        assert(isinstance(last_layer,layer))
        self.last_layer = last_layer #indicate input layer
        self.last_layer.next_layer = self
        self.fuc = fuc
        self.back_fuc = back_fuc
        self.stage = np.zeros((num_output,1))
        self.weight = np.random.rand(self.num_output,
                                     self.last_layer.num_output) /10
        self.back_error = np.zeros((1,self.last_layer.num_output)) 
        #indicate error return to last layer
    
    def load_weight(self,weight):
        assert(isinstance(weight,np.ndarray))
        assert(np.shape(weight)==np.shape(self.weight))
        self.weight = weight.copy()

    def return_weight(self):
        return self.weight

    def for_prop(self):
        self.stage = self.weight @ self.last_layer.output
        self.output = self.fuc(self.stage)
        #print('for_prop')

    def back_prop(self):
        stage_error = (self.back_fuc(self.stage) 
            * self.next_layer.back_error)
        self.back_error = self.weight.T @ stage_error
        self.weight = (self.weight + 
                       stage_error @ self.last_layer.output.T)



class output_layer(layer):
    def __init__(self, last_layer,learn_rate):
        assert(isinstance(last_layer,layer))
        self.last_layer = last_layer #indicate input layer
        self.last_layer.next_layer = self
        self.back_error = np.zeros((self.last_layer.num_output,1))
        self.learn_rate = learn_rate

    def for_prop(self):
        self.output = self.last_layer.output

    def back_prop(self,correct_output):
        assert(isinstance(correct_output,np.ndarray))
        assert(np.shape(correct_output)==np.shape(self.back_error))
        self.back_error = ((correct_output-self.output) * 
            self.learn_rate)
        return correct_output-self.output




def fuc(x):
    return x

if __name__ == "__main__":
    num1 = 10
    num2 = 5
    num3 = 6
    num4 = 1
    il = input_layer(num1)
    h1 = hidden_layer(num2,il,fuc,fuc)
    h2 = hidden_layer(num3,h1,fuc,fuc)
    h3 = hidden_layer(num4,h2,fuc,fuc)
    inp = np.random.rand(10,1)
    print(type(inp))
    il.get_input(inp)
    h1.for_prop()
    h2.for_prop()
    h3.for_prop()
    print(il.output)
    print(h1.output)
    print(h2.output)
    print(h3.output)



