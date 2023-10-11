import numpy as np
import pandas as pd

def saveLabelledData(dataset):
    npResult=np.array(dataset)
    df = pd.DataFrame(npResult)
    sheetName='mine_dynamic_opt_numEX_thetaAndFlag.txt'
    dataRoot='./labelDataset/'+sheetName
    df.to_csv(dataRoot,sep='\t',index=False, header=None)