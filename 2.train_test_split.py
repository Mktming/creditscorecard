#1
import pandas as pd
import matplotlib.pyplot as plt #导入图像库
from sklearn.model_selection import train_test_split
import seaborn as sns

#共线性热力图
data = pd.read_csv('train_clean3.csv')
corr = data.corr()#计算各变量的相关性系数
xticks = ['x0','x1','x2','x3','x4','x5','x6','x7','x8','x9','x10']#x轴标签
yticks = list(corr.index)#y轴标签
fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
sns.heatmap(corr, annot=True, cmap='rainbow', ax=ax1, annot_kws={'size': 9, 'weight': 'bold', 'color': 'blue'})#绘制相关性系数热力图
ax1.set_xticklabels(xticks, rotation=0, fontsize=10)
ax1.set_yticklabels(yticks, rotation=0, fontsize=10)
plt.show()

#分训练集和验证集
if __name__ == '__main__':
    data = pd.read_csv('train_clean3.csv')

    data['SeriousDlqin2yrs']=1-data['SeriousDlqin2yrs']
    Y = data['SeriousDlqin2yrs']
    X = data.iloc[:, 1:]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
    # print(Y_train)
    train = pd.concat([Y_train, X_train], axis=1)
    test = pd.concat([Y_test, X_test], axis=1)
    clasTest = test.groupby('SeriousDlqin2yrs')['SeriousDlqin2yrs'].count()
    train.to_csv('TrainData.csv',index=False)
    test.to_csv('TestData.csv',index=False)
    print(train.shape)
    print(test.shape)

