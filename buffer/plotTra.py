import numpy as np
import pandas
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TKAgg')

root='./labelDataset/mine_dynamic_opt_buffer_thetaAndFlag_connected100.txt'
trdat = np.array(pandas.read_csv(root, header=None,sep='\t'))
pltR=[]
for i in trdat:
    if i[-1]>0.1:
        pltR.append(i[0:-1])
pltR=np.array(pltR)
for i in pltR:
    plt.plot(i)
plt.show()
print(trdat[:,0:-1])
print(trdat[:,-1:])
