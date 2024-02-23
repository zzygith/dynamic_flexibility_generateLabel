from multiprocessing import Pool
import os, time, random
from generateLabelFuncConnected import labelFunc
import numpy as np
from saveLabelledData import saveLabelledData

if __name__=='__main__':
    #randomList=np.random.uniform(0.2,0.7,100)
    #randomList=np.arange(0.1,1,0.1)
    async_resultList=[]
    final_resultList=[]
    halfRandomNumber = 100 # generate training data
    #halfRandomNumber = 500 # generate test data
    theta0=0.5
    theta1=0.6
    theta2=0.7
    theta3=0.8
    theta4=0.6
    theta5=0.5
    theta6=0.4
    theta7=0.2
    theta8=0.6
    theta9=0.5
    thetaNormal=np.array([theta0,theta1,theta2,theta3,theta4,theta5,theta6,theta7,theta8,theta9])
    randomScalar = np.random.uniform(low=-1.0, high=1.0, size=halfRandomNumber)
    randomThetaStack=[]
    for i in randomScalar:
        randomThetaStackElement=thetaNormal+i*0.1
        randomThetaStack.append(randomThetaStackElement)
    randomThetaStack=np.array(randomThetaStack)

    print('Parent process %s.' % os.getpid())
    #p = Pool(len(randomList))
    p = Pool()
    for i in range(len(randomThetaStack)):
        #p.apply_async(labelFunc, args=(randomThetaStack[i],))
        async_result = p.apply_async(labelFunc, args=(randomThetaStack[i],))
        async_resultList.append(async_result)
    print('Waiting for all subprocesses done...')
    p.close()
    p.join()
    # result = async_result.get()
    # finalResult.append(result)
    print('All subprocesses done.')
    for i in async_resultList:
        final_resultList.append(i.get())
    print(final_resultList)
    #sheetName='mine_dynamic_opt_buffer_thetaAndFlag_connected100.txt' # generate training data
    #sheetName='mine_dynamic_opt_buffer_thetaAndFlag_connected100_2.txt' # generate training data
    #sheetName='mine_dynamic_opt_buffer_thetaAndFlag_connected100_3.txt' # generate training data
    sheetName='mine_dynamic_opt_buffer_thetaAndFlag_connected100_4.txt' # generate training data
    saveLabelledData(final_resultList,sheetName)

