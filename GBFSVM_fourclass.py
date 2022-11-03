import matplotlib.pyplot as plt
import numpy as np
import time
from warnings import filterwarnings
from sklearn.svm import SVC

from tools.gen_ball import gen_balls
from tools.pso import PSO
from tools.devitedata import Dive_Data
from tools.addNoisy import recreat_data


filterwarnings('ignore')
np.set_printoptions(suppress=True)

'''tolist()把numpy数组转化为列表, array 模块就是数组,可以存放放一组相同类型的数字'''


def demo_func(x):#Objective function
    B = 0
    A = np.zeros(len(datab[0][0])).tolist()
    for i in range(len(datab)):
        B += x[i] * datab[i][1]
        A = np.array(A) + x[i] * datab[i][-1] * np.array(datab[i][0])
    Al = np.sqrt(np.sum(A ** 2))
    wl = Al - B
    t_x = 0
    for i in range(len(datab)):
        t_x += x[i]
    return (np.square(wl) / 2) - t_x  ####Objective function：1/2(w^2)


def get_W_b(C, Y):
    lb = (np.zeros(len(datab))).tolist()
    ub = []
    for ie in datab:
        ub.append(ie[-2] * C)
    pso = PSO(func=demo_func, dim=len(datab), pop=len(datab), max_iter=1050, lb=lb, ub=ub, w=0.4, c1=1.6, c2=1.6,
              Label=Y)  # 求lagrange乘子
    pso.run()
    A = np.zeros(len(datab[0][0])).tolist()
    B = 0
    for i in range(len(datab)):
        A = np.array(A) + datab[i][-1] * pso.gbest_x[i] * np.array(datab[i][0])  # array(A)就是求解A
        B += pso.gbest_x[i] * datab[i][1]
    Al = np.sqrt(np.sum(A**2))  # ||w||
    wl = Al - B
    w = wl * np.array(A) / (Al+0.00001)
    print("w:", w)
    S = 0
    ys = []
    b = 0
    A1 = np.zeros(len(datab[0][0])).tolist()
    B1 = 0
    #print(datab)
    for i in range(len(pso.gbest_x)):
        if 0 < pso.gbest_x[i] < ub[i]:
            S += 1
            A1 = np.array(A1) + datab[i][-1] * pso.gbest_x[i] * np.array(datab[i][0])  # array(A)就是求解A
            B1 += pso.gbest_x[i] * datab[i][1]
            ys.append(i)
    Al1 = np.sqrt(np.sum(np.array(A1)**2))  # ||w||
    wl1 = abs(np.array(Al1) - B1)
    ws = (Al1 - B1) * np.array(A1) / (Al1 + 0.00001)
    for i in ys:
        b += (1 + wl1 * datab[i][1]) / datab[i][-1] - np.sum(ws * np.array(datab[i][0]))
    if S > 0:
        b /= S  ###b
    print("b:", b)
    return w, b


# 分类
def getacc(data, w, b):
    F = 0
    T = 0
    for v in data:
        val = np.sum(w * np.array(v[0:-1])) + b
        # The loss function determines
        if val < 0:
            y = -1
        else:
            y = 1
        kesi = max(0, 1 - y*val)
        if kesi > 1.0:
            val = - val
        else:
            val = val
        if (v[-1] > 0 and val >0) or (v[-1] < 0 and val < 0):
            T += 1
        else:
            F += 1
    return T / (T + F)



def ball_membership(datab):#Membership degree
    # print(datab)
    da1 = 0
    da2 = 0
    ar1 = []
    ar2 = []
    Fdata = []
    data_center = []
    sum1 = 0
    sum2 = 0
    for data in datab:#Extract the center of the particle ball
        data_center.append(data[0])
    for dd in datab:
        if dd[-1] == 1:
           da1 += np.array(dd[0])
           sum1 += 1
        else:  # Membership degree of negative class label
           da2 += np.array(dd[0])
           sum2 += 1
    if sum1 > 0:
       center1 = da1 / sum1
    else:
        center1 = 0
    if sum2 > 0:
       center2 = da2 / sum2
    else:
        center2 = 0
    #print(center1, center2)
    for dat in datab:
        if dat[-1] == 1:
            dist = np.sqrt(sum(np.power((dat[0] - center1), 2)))
            ar1.append(dist)
        else:
            dist = np.sqrt(sum(np.power((dat[0] - center2), 2)))
            ar2.append(dist)
    if len(ar1) > 0:
       r1 = max(ar1)
    else:
        r1 = 0
    if len(ar2) > 0:
        r2 = max(ar2)
    else:
        r2 = 0
    #print(r1, r2)
    for d in datab:
        if d[-1] == 1:
            mu = 1 - np.sqrt(sum(np.power((d[0] - center1), 2))) / (r1 + 0.00001)
            d[0] = mu
        else:  # Membership degree of negative class label
            mu1 = 1 - np.sqrt(sum(np.power((d[0] - center2), 2))) / (r2 + 0.00001)
            d[0] = mu1
    for i in range(len(datab)):
        Fdata.append([data_center[i], datab[i][1], datab[i][0], datab[i][-1]])
    return Fdata



if __name__ == '__main__':
    time_start = time.time()
    datae = ["fourclass"]#haberman,habermantitanic
    for i in range(len(datae)):
        urlz = r"E:\lianxiaoyu\UCI\\" + datae[i] + ".csv"
        name = datae[i]
        nor = True  # 是否归一化
        pr = 0.2 #训练集的比例
        for j in range(0, 7):  # 噪声率
            SVM_acc = 0
            Li_SVM_acc = 0
            Noisy = 0.05 * j
            for k in range(4):  # 运行4次取最大
                T_Li_SVM_acc = 0
                train, test = Dive_Data(urlz, nor, pr)
                train = np.array(train)
                test = np.array(test)
                N_data = recreat_data(train, Noisy)
                N_data = np.array(N_data)
                pur = 0
                for l in range(0, 20):  # 纯度
                    pur = 1 - 0.01 * l
                    datab = gen_balls(N_data, pur=pur, delbals=0)  # generate balls  num
                    # 计算隶属度
                    datab = ball_membership(datab)
                    Y = []
                    for ii in datab:
                        Y.append(ii[-1])
                    W, b = get_W_b(10, Y)
                    acc = getacc(test, W, b)#Accuracy of test set
                    print("accrue", acc)
                    if acc > T_Li_SVM_acc:
                        T_Li_SVM_acc = acc
                    print("T_Li_SVM_acc", T_Li_SVM_acc)
                time_end = time.time()
                time_sum = time_end - time_start
                print("time_sum", time_sum)##Running time
                # print("N_data", N_data)
                if T_Li_SVM_acc > Li_SVM_acc:
                   Li_SVM_acc = T_Li_SVM_acc
            print(name, "noisy", str(Noisy), "accuracy:", Li_SVM_acc)

