#-*-coding:utf-8-*-
import os
import cPickle
import numpy as np
from random import randrange


# 计算SVM的损失函数，以及W的梯度
# [D: 维度， N:训练样本个数，C：类别数量] 
# W:权重(D*C), X:训练样本(N*D),y:训练样本对应的label(N*1), reg：正则化系数
def loss_func(W, X, y, reg):
    # 数据初始化
    loss = 0.0
    dW = np.zeros(W.shape)  
    scores = X.dot(W)        
    num_classes = W.shape[1]
    num_train = X.shape[0]

    # 计算损失函数L = 1/N ∑Li + λR(W)
    scores_correct = scores[np.arange(num_train), y]   
    scores_correct = np.reshape(scores_correct, (num_train, -1))
    margins = scores - scores_correct + 1    
    margins = np.maximum(0,margins) # Li = ∑max(0,s_j - s_yi + △)
    margins[np.arange(num_train), y] = 0
    loss += np.sum(margins) / num_train # 1/N * ΣLi
    loss += 0.5 * reg * np.sum(W * W)   # λ = 0.5

    # 计算W的偏导
    margins[margins > 0] = 1
    row_sum = np.sum(margins, axis=1)                  
    margins[np.arange(num_train), y] = -row_sum        
    dW += np.dot(X.T, margins)/num_train + reg * W    

    return loss, dW

class SVM(object):

    def __init__(self):
        self.W = None

    # 读取文件，返回（数据，标签）
    def read_from_file(self, filename):
        with open(filename, 'rb') as f:
            datadict = cPickle.load(f)
            X = datadict['data']
            Y = datadict['labels']
            X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
            Y = np.array(Y)
        return X, Y
    
    # 读取数据集文件夹，返回(训练数据，训练标签，测试数据，测试标签)
    def read_all_file(self, ROOT):
        xs = []
        ys = []
        for b in range(1,6):
            f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
            X, Y = self.read_from_file(f)
            xs.append(X)
            ys.append(Y)    
        Xtr = np.concatenate(xs)
        Ytr = np.concatenate(ys)
        del X, Y
        Xte, Yte = self.read_from_file(os.path.join(ROOT, 'test_batch'))
        return Xtr, Ytr, Xte, Yte

    # 使用线性分类器f(x_i,W,b) = Wx_i + b进行训练，
    # 并且使用随机梯度下降的方法对W进行优化W_new = W_old - αW' (α为学习率)
    # X:训练数据， y：训练数据的标签， learing_rate：学习率α, 
    # num_iters:优化W的步长，batch_size：每个步长用到的训练数据量
    def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=100, batch_size=200):
        # 数据初始化（随机生成一个矩阵W）
        num_train, dim = X.shape
        num_classes = np.max(y) + 1  
        if self.W is None:
            self.W = 0.001 * np.random.randn(dim, num_classes)

        # 使用随机梯度下降的方法对W进行优化
        loss_history = []
        for it in xrange(num_iters):
            X_batch = None
            y_batch = None

            sample_indices = np.random.choice(np.arange(num_train), batch_size)
            X_batch = X[sample_indices]
            y_batch = y[sample_indices]

            loss, grad = loss_func(self.W,X_batch, y_batch, reg)
            loss_history.append(loss)
            print it

            self.W -= learning_rate * grad

        return loss_history

    # 使用线性分类器对结果进行预测
    def predict(self, X):
        y_pred = np.zeros(X.shape[1])
        y_pred = np.argmax(X.dot(self.W), axis=1)

        return y_pred
    

if __name__ == "__main__":
    # 加载数据集
    svm = SVM()
    data_path = "cifar-10-batches-py/"
    X_train, y_train, X_test, y_test = svm.read_all_file(data_path)
    # END

    # 初始化数据，将数据分为训练、测试两部分
    num_training = 50000
    num_test = 10000


    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]

    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]
    # END

    # 预处理：将(1*D)的图像转置成(D*1)
    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    X_test = np.reshape(X_test, (X_test.shape[0], -1))
    # END

    # 预处理：将图像的所有像素值减去均值后
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_test -= mean_image
    X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
    X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
    # END

    # 训练线性分类器，W_demo是随机生成的。
    W = np.random.randn(3073, 10) * 0.0001
    loss_hist = svm.train(X_train, y_train, learning_rate=1e-7, reg=5e4, num_iters=1500, batch_size = 1000)
    # END

    temp = svm.predict(X_test)
    print "Total: ", num_test, '\tRight rate = : %f' % (np.mean(y_test == temp), )