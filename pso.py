#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019/8/20
# @Author  : github.com/guofei9987

import numpy as np

from tools.base import SkoBase
from tools.tools import func_transformer


class PSO(SkoBase):
    """
    Do PSO (Particle swarm optimization) algorithm.

    This algorithm was adapted from the earlier works of J. Kennedy and
    R.C. Eberhart in Particle Swarm Optimization [IJCNN1995]_.

    The position update can be defined as:

    .. math::

       x_{i}(t+1) = x_{i}(t) + v_{i}(t+1)

    Where the position at the current step :math:`t` is updated using
    the computed velocity at :math:`t+1`. Furthermore, the velocity update
    is defined as:

    .. math::

       v_{ij}(t + 1) = w * v_{ij}(t) + c_{p}r_{1j}(t)[y_{ij}(t) − x_{ij}(t)]
                       + c_{g}r_{2j}(t)[\hat{y}_{j}(t) − x_{ij}(t)]

    Here, :math:`cp` and :math:`cg` are the cognitive and social parameters
    respectively. They control the particle's behavior given two choices: (1) to
    follow its *personal best* or (2) follow the swarm's *global best* position.
    Overall, this dictates if the swarm is explorative or exploitative in nature.
    In addition, a parameter :math:`w` controls the inertia of the swarm's
    movement.

    .. [IJCNN1995] J. Kennedy and R.C. Eberhart, "Particle Swarm Optimization,"
    Proceedings of the IEEE International Joint Conference on Neural
    Networks, 1995, pp. 1942-1948.

    Parameters
    --------------------
    func : function
        The func you want to do optimal
    dim : int
        Number of dimension, which is number of parameters of func.维数，即函数的参数数
    pop : int
        Size of population, which is the number of Particles. We use 'pop' to keep accordance with GA 种群大小，即粒子的数量。我们使用“pop”来保持符合GA
    max_iter : int
        Max of iter iterations  迭代器迭代次数上限
    lb : array_like
        The lower bound of every variables of func函数的每个变量的下限
    ub : array_like
        The upper bound of every variables of func函数的每个变量的上限
    constraint_eq : tuple
        equal constraint. Note: not available yet.相等的约束。注意：尚不可用
    constraint_ueq : tuple
        unequal constraint不等约束
    Attributes属性
    ----------------------
    pbest_x : array_like, shape is (pop,dim)
        best location of every particle in history历史上每个粒子的最佳位置
    pbest_y : array_like, shape is (pop,1)
        best image of every particle in history历史上每个粒子的最佳图像
    gbest_x : array_like, shape is (1,dim)
        general best location for all particles in history历史上所有粒子的一般最佳位置
    gbest_y : float
        general best image  for all particles in history历史上所有粒子的一般最佳图像
    gbest_y_hist : list
        gbest_y of every iteration每次迭代的gbest_y


    Examples
    -----------------------------
    see https://scikit-opt.github.io/scikit-opt/#/en/README?id=_3-psoparticle-swarm-optimization
    """

    def __init__(self, func, n_dim=None, pop=40, max_iter=150, lb=-1e5, ub=1e5, w=0.8, c1=0.5, c2=0.5,
                 constraint_eq=tuple(), constraint_ueq=tuple(), verbose=False
                 , dim=None, Label=[]):

        n_dim = n_dim or dim  # support the earlier version

        self.func = func_transformer(func)
        self.w = w  # inertia
        self.Label = Label
        self.cp, self.cg = c1, c2  # parameters to control personal best, global best respectively分别控制个人最佳、全球最佳的参数
        self.pop = pop  # number of particles粒子数
        self.n_dim = n_dim  # dimension of particles, which is the number of variables of func粒子的维度，即func的变量个数
        self.max_iter = max_iter  # max iter
        self.verbose = verbose  # print the result of each iter or not

        self.lb, self.ub = np.array(lb) * np.ones(self.n_dim), np.array(ub) * np.ones(self.n_dim)
        assert self.n_dim == len(self.lb) == len(self.ub), 'dim == len(lb) == len(ub) is not True'#assert（断言）用于判断一个表达式，在表达式条件为 false 的时候触发异常
        assert np.all(self.ub > self.lb), 'upper-bound must be greater than lower-bound 上限必须大于下限'

        self.has_constraint = bool(constraint_ueq)
        self.constraint_ueq = constraint_ueq
        self.is_feasible = np.array([True] * pop)#可行的

        # self.X = np.random.uniform(low=self.lb, high=self.ub, size=(self.pop, self.n_dim))
        self.X = self.ini_X()

        v_high = self.ub - self.lb
        self.V = np.random.uniform(low=-v_high, high=v_high, size=(self.pop, self.n_dim))  # speed of particles粒子速度
        self.Y = self.cal_y()  # y = f(x) for all particles
        self.pbest_x = self.X.copy()  # personal best location of every particle in history
        self.pbest_y = np.array([[np.inf]] * pop)  # best image of every particle in history历史上每个粒子的最佳图像
        self.gbest_x = self.pbest_x.mean(axis=0).reshape(1, -1)  # global best location for all particles
        self.gbest_y = np.inf  # global best y for all particles
        self.gbest_y_hist = []  # gbest_y of every iteration
        self.update_gbest()

        # record verbose values
        self.record_mode = False
        self.record_value = {'X': [], 'V': [], 'Y': []}
        self.best_x, self.best_y = self.gbest_x, self.gbest_y  # history reasons, will be deprecated
    def ini_X(self):
        # X=np.random.uniform(low=self.lb, high=self.ub, size=(self.pop, self.n_dim))
        # total = 0
        # total1 = 0
        Z = np.random.uniform(low=self.lb, high=self.ub, size=(self.pop, self.n_dim))  # 10为粒子个数，3位粒球个数
        Label = self.Label
        re = np.zeros((self.pop, 1))
        re1 = np.zeros((self.pop, 1))
        for j in range(self.n_dim):
            if Label[j] > 0:
                re += Z[:, [j]]
            else:
                re1 += Z[:, [j]]
        mul = 0.
        sa = np.sum(re)
        sa1 = np.sum(re1)
        if sa > sa1:
            mul = sa1 / sa
            for j in range(self.n_dim):
                if Label[j] > 0:
                    Z[j] *= mul
        else:
            mul = sa / sa1
            for j in range(self.n_dim):
                if Label[j] < 0:
                    Z[j] *= mul
        return Z

    def check_constraint(self, x):
        # gather all unequal constraint functions收集所有不等约束函数
        # print('self.constraint_ueq', self.constraint_ueq)and constraint_func(x) < self.ub1
        for constraint_func in self.constraint_ueq:
            if constraint_func(x) > 0:
                return False
        return True

    def update_V(self):##更新自己的速度
        r1 = np.random.rand(self.pop, self.n_dim)
        r2 = np.random.rand(self.pop, self.n_dim)
        self.V = self.w * self.V + \
                 self.cp * r1 * (self.pbest_x - self.X) + \
                 self.cg * r2 * (self.gbest_x - self.X)

    def update_X(self):##更新位置
        self.X = self.X + self.V
        self.X = np.clip(self.X, self.lb, self.ub)##越界修改
        ####更新位置的约束α*Y=0
        Label = self.Label
        re = np.zeros((self.pop, 1))
        re1 = np.zeros((self.pop, 1))
        for j in range(self.n_dim):
            if Label[j] > 0:
                re += self.X[:, [j]]
            else:
                re1 += self.X[:, [j]]
        mul = 0.
        sa = np.sum(re)
        sa1 = np.sum(re1)
        if sa > sa1:
            mul = sa1 / sa
            for j in range(self.n_dim):
                if Label[j] > 0:
                    self.X[j] *= mul
        else:
            mul = sa / sa1
            for j in range(self.n_dim):
                if Label[j] < 0:
                    self.X[j] *= mul







    def cal_y(self):
        # calculate y for every x in X
        self.Y = self.func(self.X).reshape(-1, 1)#固定成1列，行数系统自动生成
        return self.Y

    def update_pbest(self):##个人最好的位置
        '''
        personal best
        :return:
        '''
        self.need_update = self.pbest_y > self.Y #判断位置是否是大于Y的，是的话返回Ture，否返回False
        for idx, x in enumerate(self.X):
            if self.need_update[idx]:
                self.need_update[idx] = self.check_constraint(x)

        self.pbest_x = np.where(self.need_update, self.X, self.pbest_x)###那么，当condition中的值是true时返回x对应位置的值，false是返回y的。
        self.pbest_y = np.where(self.need_update, self.Y, self.pbest_y)

    def update_gbest(self):##全局最优
        '''
        global best
        :return:
        '''
        idx_min = self.pbest_y.argmin()
        if self.gbest_y > self.pbest_y[idx_min]:
            self.gbest_x = self.X[idx_min, :].copy()
            self.gbest_y = self.pbest_y[idx_min]

    def recorder(self):
        if not self.record_mode:
            return
        self.record_value['X'].append(self.X)
        self.record_value['V'].append(self.V)
        self.record_value['Y'].append(self.Y)

    def run(self, max_iter=None, precision=1e-7, N=20):
        '''
        precision: None or float
            If precision is None, it will run the number of max_iter steps
            If precision is a float, the loop will stop if continuous N difference between pbest less than precision
        N: int
        '''
        self.max_iter = max_iter or self.max_iter
        c = 0
        # old_gbest_y = 0
        for iter_num in range(self.max_iter):
            self.update_V()
            self.recorder()
            self.update_X()
            self.cal_y()#目标函数
            self.update_pbest()#个人最优
            self.update_gbest()#全局最优
            #判断循环是否结束
            if precision is not None:
                tor_iter = np.amax(self.pbest_y) - np.amin(self.pbest_y)
                if tor_iter < precision:
                    c = c + 1
                    if c > N:
                        break
                else:
                    c = 0
                # tor_iter = abs(self.gbest_y - old_gbest_y)
                # old_gbest_y = self.gbest_y
                # if tor_iter < precision:
                #     c += 1
                #     if c > N:
                #         break
                # else:
                #     c = 0
            #是否打印出每一步的结果
            if self.verbose:
                print('Iter: {}, Best fit: {} at {}'.format(iter_num, self.gbest_y, self.gbest_x))

            self.gbest_y_hist.append(self.gbest_y)
        self.best_x, self.best_y = self.gbest_x, self.gbest_y
        return self.best_x, self.best_y

    fit = run
