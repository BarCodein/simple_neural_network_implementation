from read import read_from_csv
from net import net
from fuc import Relu,de_Relu,sigmod,de_sigmod
import numpy as np
from operation import train,test
train_data,tag = read_from_csv("train.csv")
print('td',train_data)

n = net([7,5,5,5,1]
        ,[Relu,Relu,Relu,sigmod],0.05)
n.load_weight("t.npz")
#n.layers[1].load_weight(np.random.rand(5,10))
for j in range(50):
    er = train(n,train_data[:800],tag[:800])
    print(er)

print(test(n,train_data[800:],tag[800:]))

n.store_weight("t.npz")
    


