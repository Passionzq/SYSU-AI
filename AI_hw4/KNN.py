#-*-coding:utf-8-*-
from __future__ import division
import numpy as np
import cPickle
import time

total = 10


def cmp(x):
    return x[0]

class KNN(object):
    
    def __init__(self):
        self.imgs = []
        self.labels = []
        self.k = 5

    # 读取文件，返回一个字典
    def read_from_file(self, filename):
        with open(filename, 'rb') as fo:
            dict = cPickle.load(fo)
        return dict
    
    # KNN的训练过程就是将训练集以及相应的标签
    def train(self):
        for i in range (1,6):
            filename = "./cifar-10-batches-py/data_batch_" + str(i)
            dict = self.read_from_file(filename)
            for j in range(0,len(dict['labels'])):
                self.imgs.append(dict['data'][j])
                self.labels.append(dict['labels'][j])
                j += 1
    
    def predict(self):
        with open("./cifar-10-batches-py/test_batch", 'rb') as fo:
            dict = cPickle.load(fo)
        test_imgs = dict['data']
        test_labels = dict['labels']
        predict_labels = []
        is_true = []

        global total

        start = time.time()

        for i in range (0, total):
            dist = self.cal_L2(test_imgs[i])
            
            vote = [0,0,0,0,0,0,0,0,0,0]
            for d in dist:
                vote[d[1]] += 1
            
            temp = [-1, 0]
            for j in range(10):
                if vote[j] > temp[0]:
                    temp[0] = vote[j]
                    temp[1] = j

            predict_labels.append(temp[1])

            if temp[1] == test_labels[i]:
                is_true.append(1)
            else:
                is_true.append(0)
        
        end = time.time()
        
        # print predict_labels
        # print test_labels[:total]
        print "Cost time = ", end - start, "s"
        print "[Distance: L2] K =", self.k,"\tRight =", sum(is_true), "\tTotal =", total, "\tRight rate = ", sum(is_true) / total,"\n"
        

    # 计算L1距离，返回前K个最小距离的label
    def cal_L1(self, input_img):
        distances = []
        for i in range (0, len(self.imgs)):
            distances.append([np.sum(np.abs(input_img - self.imgs[i])), self.labels[i]])

        distances.sort(key = cmp)

        return distances[:self.k]    
    

    # 计算L2距离，返回前K个最小距离的label
    def cal_L2(self, input_img):
        distances = []
        for i in range (0, len(self.imgs)):
            distances.append([np.sqrt(np.sum(np.square(input_img - self.imgs[i]))), self.labels[i]])

        distances.sort(key = cmp)

        return distances[:self.k]

        

if __name__ == "__main__":
    knn = KNN()
    knn.train()
    totals = [10, 100, 1000, 10000]
    for i in totals:
        total = i
        knn.predict()

