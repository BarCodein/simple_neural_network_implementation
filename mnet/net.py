import numpy as np
from layer import layer,input_layer,hidden_layer,output_layer

class net:
    def __init__(self,nodes_num,fucs):
        assert(isinstance(nodes_num,list))
        assert(isinstance(fucs,list))
        assert(len(nodes_num)==len(fucs)+1)
        self.layers = []
        self.layers.append(input_layer(nodes_num[0]))
        for i in range(1,len(nodes_num)):
            self.layers.append(hidden_layer(nodes_num[i]
                ,self.layers[-1],fucs[i-1]))
        self.layers.append(output_layer(self.layers[-1]))
    
    def for_prop(self,inp):
        self.layers[0].get_input(inp)
        for i in range(1,len(self.layers)):
            self.layers[i].for_prop()

    def print_output(self):
        print(self.layers[-1].output)

def a(x):
    return x

if __name__ == "__main__":
    n = net([10,5,6,3,4]
            ,[a,a,a,a])
    inp = np.random.rand(10,1)
    n.for_prop(inp)
    n.print_output()