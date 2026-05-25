from read import read_from_csv
from net import net
from fuc import Relu,de_Relu,sigmod,de_sigmod
import numpy as np

train_data,tag = read_from_csv()
print('td',train_data)

n = net([7,5,3,4,1]
        ,[Relu,Relu,Relu,sigmod],
        [de_Relu,de_Relu,de_Relu,de_sigmod]
        ,0.01)
#n.layers[1].load_weight(np.random.rand(5,10))
inp = train_data[[0]].T
n.for_prop(inp)
n.print_output()
oup = tag[[0]]
n.back_prop(oup)

for i in range(800):
    inp = train_data[[i]].T
    n.for_prop(inp)
    oup = tag[[i]]
    err = n.train(inp,oup)
    if i%10==0:
        print('value',round(n.get_output()[0][0],3))
        print('error',abs(round(err[0][0],3)))

    n.store_weight("t.npz")
    


