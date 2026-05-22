from net import net,a,b
import numpy as np
n = net([10,5,6,3,4]
            ,[a,a,a,a],[b,b,b,b],0.01)
n.load_weight("w.npz")
inp = np.ones((10,1))
n.for_prop(inp)
n.print_output()
oup = np.array([[1],[0],[1],[0]]) 
n.back_prop(oup)
for i in range(800):
    n.for_prop(inp)
    if i%10==0:
        n.print_output()
    n.back_prop(oup)
n.store_weight("w.npz")
    



