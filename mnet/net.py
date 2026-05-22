import numpy as np
from layer import layer,input_layer,hidden_layer,output_layer

class net:
    def __init__(self,nodes_num,fucs,back_fucs,learn_rate):
        assert(isinstance(nodes_num,list))
        assert(isinstance(fucs,list))
        assert(len(nodes_num)==len(fucs)+1)
        assert(len(nodes_num)==len(back_fucs)+1)
        self.layers = []
        self.layers.append(input_layer(nodes_num[0]))
        for i in range(1,len(nodes_num)):
            self.layers.append(hidden_layer(nodes_num[i]
                ,self.layers[-1],fucs[i-1],back_fucs[i-1]))
        self.layers.append(output_layer(self.layers[-1],learn_rate))
    
    def load_weight(self,filename):
        data = np.load(filename)
        for key in data:
            self.layers[int(key)].load_weight(data[key])

    def store_weight(self,filename):
        weights = {}
        for i in range(1,len(self.layers)-1):
            weights[str(i)] = self.layers[i].return_weight()
        np.savez(filename,**weights)



    def for_prop(self,inp):
        self.layers[0].get_input(inp)
        for i in range(1,len(self.layers)):
            self.layers[i].for_prop()

    def back_prop(self,correct_output):
        self.layers[-1].back_prop(correct_output)
        for i in range(len(self.layers)-2,0,-1):
            self.layers[i].back_prop()


    def print_output(self):
        print(self.layers[-1].output)

def a(x):
    return np.where(x>0,x,0)

def b(error,value):
    t =  np.where(value>0,1,0)
    return t*error

if __name__ == "__main__":
    n = net([10,5,6,3,4]
            ,[a,a,a,a],[b,b,b,b],0.01)
    #n.layers[1].load_weight(np.random.rand(5,10))
    inp = np.ones((10,1))
    n.for_prop(inp)
    n.print_output()
    oup = np.array([[1],[0],[2],[1]]) 
    n.back_prop(oup)

    for i in range(1500):
        n.for_prop(inp)
        if i%10==0:
            n.print_output()
        n.back_prop(oup)
    n.store_weight("w.npz")
    



