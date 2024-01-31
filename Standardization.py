
import numpy as np
import pandas as pd
from sklearn import preprocessing

#read
def Standardization_excel(filepath):
    df=[]
    df=pd.read_excel(filepath,sheet_name='Sheet1',header=None).values
    lables = df[:,0:1]
    # print(lables)
    df = df[:,1:].T
    # df=preprocessing.MinMaxScaler().fit_transform(df)
    df = preprocessing.StandardScaler().fit_transform(df)
    return df,lables

def Standardization_csv(filepath):
    df=[]
    df=pd.read_csv(filepath,delimiter=',',header=None)
    df=preprocessing.MinMaxScaler().fit_transform(df)
    return df
print('new')
# filepath=r'./data_spec/exp2/train.xlsx'
# filepath=r'D:\mysoftware\py37\program\Arthritis\data\second-data（500-2500)/Smooth-zong-airpls.xlsx'
filepath=r'D:\mysoftware\py37\program\Arthritis\data\second-data（500-2500)/Smooth-Arthritis_zong_data(500-2500)-airpls.xlsx'
# filepath=r'D:\mysoftware\py37\program\Arthritis\data\second-data（500-2500)/Smoothing-Arthritis_zong_data(500-2500)-airpls.xlsx'
file=Standardization_excel(filepath)
df,lable=file
print(lable.shape)
# print(df.shape)
df = df.T
print(df.shape)
df = np.hstack((lable,df))
df = pd.DataFrame(df)
# df.to_excel(r'./data_spec/exp2/train健康归一化.xlsx',header=None,index=None)
# df.to_excel(r'D:\mysoftware\py37\program\Arthritis\data\second-data（500-2500)/Smooth-gui-zong-airpls.xlsx',header=None,index=None)
df.to_excel(r'D:\mysoftware\py37\program\Arthritis\data\second-data（500-2500)/Smooth-gui1-Arthritis_zong_data(500-2500)-airpls.xlsx',header=None,index=None)
# df.to_excel(r'D:\mysoftware\py37\program\Arthritis\data\second-data（500-2500)/Smoothing-Arthritis_zong_data(500-2500)-airpls-guiyihua.xlsx',header=None,index=None)
