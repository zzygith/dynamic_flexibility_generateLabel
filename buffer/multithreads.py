from multiprocessing import Pool
import os, time, random
from generateLabelFunc import labelFunc
import numpy as np
from saveLabelledData import saveLabelledData


# def long_time_task(name):
#     print('Run task %s (%s)...' % (name, os.getpid()))
#     start = time.time()
#     time.sleep(random.random() * 3)
#     end = time.time()
#     print('Task %s runs %0.2f seconds.' % (name, (end - start)))



if __name__=='__main__':
    #randomList=np.random.uniform(0.2,0.7,100)
    #randomList=np.arange(0.1,1,0.1)
    async_resultList=[]
    final_resultList=[]
    halfRandomNumber = 50 # generate training data
    #halfRandomNumber = 500 # generate test data
    theta0_min = [0.3, 0.9]
    theta0_max = [0.4, 1]
    randomTheta0_1 = np.random.uniform(low=theta0_min[0], high=theta0_max[0], size=(halfRandomNumber,1))
    randomTheta0_2 = np.random.uniform(low=theta0_min[1], high=theta0_max[1], size=(halfRandomNumber,1))
    randomTheta0Stack=np.vstack((randomTheta0_1,randomTheta0_2))
    np.random.shuffle(randomTheta0Stack)
    theta1_min = [0.8]
    theta1_max = [0.9]
    randomTheta1 = np.random.uniform(low=theta1_min[0], high=theta1_max[0], size=(2*halfRandomNumber,1))

    theta2_min = [0.5, 0.9]
    theta2_max = [0.6, 1]

    randomTheta2_1 = np.random.uniform(low=theta2_min[0], high=theta2_max[0], size=(halfRandomNumber,1))
    randomTheta2_2 = np.random.uniform(low=theta2_min[1], high=theta2_max[1], size=(halfRandomNumber,1))
    randomTheta2Stack=np.vstack((randomTheta2_1,randomTheta2_2))
    np.random.shuffle(randomTheta2Stack)

    theta3_min = [0.6]
    theta3_max = [0.7]
    randomTheta3 = np.random.uniform(low=theta3_min[0], high=theta3_max[0], size=(2*halfRandomNumber,1))
    randomThetaStack=np.hstack((randomTheta0Stack,randomTheta1,randomTheta2Stack,randomTheta3))

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
    sheetName='mine_dynamic_opt_buffer_thetaAndFlag_100.txt' # generate training data
    #sheetName='mine_dynamic_opt_buffer_thetaAndFlag_ForTest.txt' # generate test data
    saveLabelledData(final_resultList,sheetName)

#labelFunc(0.5)