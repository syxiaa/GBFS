import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import csv

def Dive_Data(urlz, nor, pr):
	"""
	:param urlz: 需要处理的数据集地址
	:param name: 存储的训练集测试机name
	:param nor:是否归一化
	:param pr:测试集合所占比例
	:param pre:噪声所占比例
	:return:
	"""
	df = pd.read_csv(urlz, header=None)
	data = df.values
	numberSample, numberAttribute = data.shape

	if nor==True:
		minMax = MinMaxScaler()  # 将数据进行归一化
		U = np.hstack((minMax.fit_transform(data[0:numberSample, 1:]), data[0:numberSample, 0].reshape(numberSample, 1)))
	else:
		U = np.hstack(((data[0:numberSample, 1:]), data[0:numberSample, 0].reshape(numberSample, 1)))

	for i in range(len(U)):
		if U[i][-1] != 1:
			U[i][-1] = -1
	np.random.shuffle(U)#shuffle ()函数是将列表的所有元素随机排序。
	U = U.tolist()
	train=U[0:int(numberSample*(1-pr))]
	test=U[int(numberSample * (1-pr)):]
	return train,test
	"""
	url_train="D:\py\粒球SVM_精度\划分后数据\\" + name +"train.csv"
	with open(url_train, "w", newline='', encoding="utf-8") as jg:
		cw = csv.writer(jg)
		cw.writerows(U[0:int(numberSample*(1-pr))])
	url_test = "D:\py\粒球SVM_精度\划分后数据\\" + name +"test.csv"
	with open(url_test, "w", newline='', encoding="utf-8") as jg:
		cw = csv.writer(jg)
		cw.writerows(U[int(numberSample * (1-pr)):])
	"""
"""
urlz="D:\\py\粒球SVM_精度\数据\\sonar.csv"
name="sonar"
nor=True
pr=0.2
Dive_Data(urlz, name, nor, pr)
"""
# print("粒球：",(0.755+0.735+0.72+0.655)/4)
# print("SVC：",(0.75+0.785+0.74+0.69)/4)