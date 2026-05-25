from net import net
from fuc import Relu,sigmod
import numpy as np
n = net([4,5,4]
        ,[Relu,sigmod],0.1)
#n.layers[1].load_weight(np.random.rand(5,10))
inp = np.array([[[10],[2],[3],[-1]],
                [[1],[2],[4],[1]]])
n.for_prop(inp)
n.print_output()
oup = np.array([[1],[0],[1],[0]]) 
n.back_prop(oup)

for i in range(1500):
    err = n.train(inp,oup)
    if i%10==0:
        n.print_output()
        print('error',err)
n.store_weight("w.npz")

