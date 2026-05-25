from net import net
from read import read_from_csv

def train(n,train_data,tag):
    s=0
    for i in range(train_data.shape[0]):
        inp = train_data[[i]].T
        oup = tag[[i]]
        err = n.train(inp,oup)
        s+= abs(err[0][0])
       # if i%10==0:
    #        print('value',round(n.get_output()[0][0],3))
        #    print('error',abs(round(err[0][0],3)))
    return s/train_data.shape[0]

def test(n,data,tag):
    s = 0
    for i in range(data.shape[0]):
        inp = data[[i]].T
        oup = tag[[i]]
        err = n.test(inp,oup)
        s+= abs(err[0][0])
    return s/data.shape[0]
 
