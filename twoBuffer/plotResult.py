import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas

# matplotlib.use('TKAgg')
# sheetNametoSave='mine_dynamic_opt_twobuffer_thetaAndFlag_10000(i55_b46).txt'
# #sheetNametoSave='mine_dynamic_opt_twobuffer_thetaAndFlag_10000(i46_b46).txt'


# root='./labelDataset/'+sheetNametoSave
# allTheta = np.array(pandas.read_csv(root, header=None,sep='\t'))
# feasibleTheta=[]
# infeasibleTheta=[]
# for i in allTheta:
#     if i[2]==1:
#         feasibleTheta.append(i)
#     else:
#         infeasibleTheta.append(i)
# feasibleTheta=np.array(feasibleTheta)        
# plt.scatter(x=feasibleTheta[:,0:1],y=feasibleTheta[:,1:2],s=1)
# infeasibleTheta=np.array(infeasibleTheta)        
# plt.scatter(x=infeasibleTheta[:,0:1],y=infeasibleTheta[:,1:2],s=1)
# plt.show()




######################
matplotlib.use('TKAgg')
#sheetNametoSave='mine_dynamic_opt_twobuffer_thetaAndFlag_10000(i55_b46).txt'
#sheetNametoSave='mine_dynamic_opt_twobuffer_thetaAndFlag_10000(i46_b46).txt'
sheetNametoSave='mine_dynamic_opt_twobuffer_thetaAndFlag_10000(b37).txt'
#sheetNametoSave='mine_dynamic_opt_twobuffer_thetaAndFlag_10000(b46).txt'

root='./labelDataset/'+sheetNametoSave
allTheta = np.array(pandas.read_csv(root, header=None,sep='\t'))
feasibleTheta=[]
infeasibleTheta=[]
for i in allTheta:
    if i[12]==1:
        feasibleTheta.append(i)
    else:
        infeasibleTheta.append(i)
feasibleTheta=np.array(feasibleTheta)        
plt.scatter(x=feasibleTheta[:,0:1],y=feasibleTheta[:,1:2],s=1)
infeasibleTheta=np.array(infeasibleTheta)        
plt.scatter(x=infeasibleTheta[:,0:1],y=infeasibleTheta[:,1:2],s=1)
plt.show()

plt.scatter(x=feasibleTheta[:,2:3],y=feasibleTheta[:,3:4],s=1)    
plt.scatter(x=infeasibleTheta[:,2:3],y=infeasibleTheta[:,3:4],s=1)
plt.show()

plt.scatter(x=feasibleTheta[:,4:5],y=feasibleTheta[:,5:6],s=1)    
plt.scatter(x=infeasibleTheta[:,4:5],y=infeasibleTheta[:,5:6],s=1)
plt.show()

plt.scatter(x=feasibleTheta[:,6:7],y=feasibleTheta[:,7:8],s=1)    
plt.scatter(x=infeasibleTheta[:,6:7],y=infeasibleTheta[:,7:8],s=1)
plt.show()

plt.scatter(x=feasibleTheta[:,8:9],y=feasibleTheta[:,9:10],s=1)    
plt.scatter(x=infeasibleTheta[:,8:9],y=infeasibleTheta[:,9:10],s=1)
plt.show()

plt.scatter(x=feasibleTheta[:,10:11],y=feasibleTheta[:,11:12],s=1)    
plt.scatter(x=infeasibleTheta[:,10:11],y=infeasibleTheta[:,11:12],s=1)
plt.show()