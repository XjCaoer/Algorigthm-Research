# -*- coding: utf-8 -*-
"""
Created on Tue May 29 10:20:11 2018

@author: Cao
"""
import random
import pandas as pd
class GA():
    def __init__(self, length, count, X_train, X_test, y_train, y_test,coding):
        # 数据
        self.X_train = X_train
        
        self.X_test = X_test
        
        self.y_train = y_train
        
        self.y_test = y_test
        # 染色体长度
        self.length = length
        # 种群中的染色体数量
        self.count = count
        #
        self.coding = coding
        # 随机生成初始种群
        self.population = self.gen_population(length, count)
        
    def evolve(self, retain_rate = 0.2, random_select_rate=0.5, mutation_rate=0.01):
        """
        进化
        对当前一代种群依次进行选择、交叉并生成新一代种群，然后对新一代种群进行变异
        """
        parents = self.selection(retain_rate, random_select_rate)
        self.crossover(parents)
        self.mutation(mutation_rate)

    def gen_chromosome(self, length):
        """
        随机生成长度为length的染色体，每个基因的取值是0或1
        这里用一个bit表示一个基因
        """
        chromosome = 0
        for i in range(length):
            #利用掩码进行位操作
            chromosome |= (1 << i) * random.randint(0, 1)
        return chromosome

    def gen_population(self, length, count):
        """
        获取初始种群（一个含有count个长度为length的染色体的列表）
        """
        return [self.gen_chromosome(length) for i in range(count)]

    def selection(self, retain_rate, random_select_rate):
        """
        选择
        先对适应度从大到小排序，选出存活的染色体
        再进行随机选择，选出适应度虽然小，但是幸存下来的个体
        """
        print("fitness")
        # 对适应度从大到小进行排序
        graded = [(self.fitness(chromosome), chromosome) for chromosome in self.population]
        graded = [x[1] for x in sorted(graded, reverse=True)]
        print("population selection")
        # 选出适应性强的染色体
        retain_length = int(len(graded) * retain_rate)
        parents = graded[:retain_length]
        # 选出适应性不强，但是幸存的染色体
        for chromosome in graded[retain_length:]:
            if random.random() < random_select_rate:
                parents.append(chromosome)
        return parents

    def crossover(self, parents):
        """
        染色体的交叉、繁殖，生成新一代的种群
        """
        print("crossover")
        # 新出生的孩子，最终会被加入存活下来的父母之中，形成新一代的种群。
        children = []
        # 需要繁殖的孩子的量
        target_count = len(self.population) - len(parents)
        # 开始根据需要的量进行繁殖
        while len(children) < target_count:
            male = random.randint(0, len(parents)-1)
            female = random.randint(0, len(parents)-1)
            if male != female:
                # 随机选取交叉点
                cross_pos = random.randint(0, self.length)
                # 生成掩码，方便位操作
                mask = 0
                for i in range(cross_pos):
                    mask |= (1 << i)
                male = parents[male]
                female = parents[female]
                # 孩子将获得父亲在交叉点前的基因和母亲在交叉点后（包括交叉点）的基因
                child = ((male & mask) | (female & ~mask)) & ((1 << self.length) - 1)
                children.append(child)
        # 经过繁殖后，孩子和父母的数量与原始种群数量相等，在这里可以更新种群。
        self.population = parents + children

    def mutation(self, rate):
        """
        变异
        对种群中的所有个体，随机改变某个个体中的某个基因
        """
        print("muation")
        for i in range(len(self.population)):
            if random.random() < rate:
                j = random.randint(0, self.length-1)
                self.population[i] ^= 1 << j

    def result(self):
        """
        获得当前代的最优值，这里取的是函数取最大值时x的值。
        """
        graded = [(self.fitness(chromosome), chromosome) for chromosome in self.population]
        graded = [(x[0],bin(x[1])) for x in sorted(graded, reverse=True)]
        return graded[0]

    def fitness(self, chromosome):
        y_train = self.y_train
        y_test = self.y_test
        X_train,X_test = self.read_data(chromosome)
        M = eval(self.coding)(X_train,y_train)
        Code_matrix = M[0]
        # get SVM classifier object.
        estimator = get_base_clf('SVM')  
        sec = SimpleECOCClassifier(estimator, Code_matrix)
        sec.fit(X_train,y_train)
        pred_label = sec.predict(X_test)
        result = Evaluation(y_test,pred_label).evaluation(accuracy=True,precision=True,sensitivity=True,Fscore=True)
        return result['accuracy']

    def read_data(self,chromosome):
        '''
        按染色体编码选择特征读取数据
        '''
        X_train = self.X_train.copy()
        X_test = self.X_test.copy()
        a = bin(chromosome)[2:]
        a = a.zfill(self.length)
        '''
        将未选中特征数据删除
        '''
        tmp = 0
        test = pd.DataFrame()
        train = pd.DataFrame()
        for i in a:
            if i == '1':
                train[train.shape[1]] = X_train[X_train.columns[tmp]]
                test[test.shape[1]] = X_test[X_test.columns[tmp]]
            tmp += 1          
        return train,test