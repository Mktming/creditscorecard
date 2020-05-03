#utf-8  2020-04-28 20:35:44
#解决数据不平衡问题
import pandas as pd
import numpy as np
import imblearn
from imblearn.over_sampling import SMOTE

data=pd.read_csv('train_clean2.csv',index_col=0)
print(data.shape)

x=data.iloc[:,1:]
y=data.iloc[:,0]
print(y.value_counts())

n_sample=x.shape[0]
n_1_sample=y.value_counts()[1]
n_0_sample=y.value_counts()[0]
print('Samples：{}; 1 {:.2%}; 0 {:.2%}'.format(n_sample,n_1_sample/n_sample,n_0_sample/n_sample))

sm = SMOTE(random_state=42) #实例化
x,y = sm.fit_sample(x,y)
n_sample_ = x.shape[0]
pd.Series(y).value_counts()
n_1_sample = pd.Series(y).value_counts()[1]
n_0_sample = pd.Series(y).value_counts()[0]
print('Samples：{}; 1 {:.2%}; 0 {:.2%}'.format(n_sample_,n_1_sample/n_sample_,n_0_sample/n_sample_))

x=pd.DataFrame(x)
y=pd.DataFrame(y)
new_data=pd.concat([y,x],axis=1)
new_data.columns=data.columns
new_data.shape

new_data.to_csv('./train_clean3.csv',index=False,columns=new_data.columns)