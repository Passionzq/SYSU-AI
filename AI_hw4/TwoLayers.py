#-*-coding:utf-8-*-
import os
import cPickle
import numpy as np
from random import randrange


class Softmax(object):

    def __init__(self):
        self.h = 100   # 隐藏层大小
        self.D = 32 * 32 * 3 # 每张图片的维数
        self.K = 10 # 一共分为10类
        self.W = 0.01 * np.random.randn(self.D,self.h)   
        self.b = np.zeros((1,self.h))
        self.W2 = 0.01 * np.random.randn(self.h,self.K)
        self.b2 = np.zeros((1,self.K))
        

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

    def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=100):
        # 使用随机梯度下降的方法对W,b进行优化
        num_examples = X.shape[0]
        for i in xrange(num_iters):
        

            hidden_layer = np.maximum(0, np.dot(X, self.W) + self.b) 
            scores = np.dot(hidden_layer, self.W2) + self.b2


            exp_scores = np.exp(scores)
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) 

            correct_logprobs = -np.log(probs[range(num_examples),y])
            data_loss = np.sum(correct_logprobs)/num_examples
            reg_loss = 0.5*reg*np.sum(self.W*self.W) + 0.5*reg*np.sum(self.W2*self.W2)
            loss = data_loss + reg_loss

            dscores = probs
            dscores[range(num_examples),y] -= 1
            dscores /= num_examples

            dW2 = np.dot(hidden_layer.T, dscores)
            db2 = np.sum(dscores, axis=0, keepdims=True)
            
            dhidden = np.dot(dscores, self.W2.T)
            
            dhidden[hidden_layer <= 0] = 0
            
            dW = np.dot(X.T, dhidden)
            db = np.sum(dhidden, axis=0, keepdims=True)
            
            dW2 += reg * self.W2
            dW += reg * self.W

            self.W += -learning_rate * dW
            self.b += -learning_rate * db
            self.W2 += -learning_rate * dW2
            self.b2 += -learning_rate * db2

    # 使用线性分类器对结果进行预测
    def predict(self, X):
        hidden_layer = np.maximum(0, np.dot(X, self.W) + self.b)
        scores = np.dot(hidden_layer, self.W2) + self.b2
        predicted_class = np.argmax(scores, axis=1)

        return predicted_class
    

if __name__ == "__main__":
    # 加载数据集
    softmax = Softmax()
    data_path = "cifar-10-batches-py/"
    X_train, y_train, X_test, y_test = softmax.read_all_file(data_path)
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

    # 预处理：将图像的所有像素值减去均值后归一化到[-1.1]
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_test -= mean_image
    X_train = np.hstack([X_train, np.ones((X_train.shape[0], 0))])
    X_test = np.hstack([X_test, np.ones((X_test.shape[0], 0))])
    # END

    softmax.train(X_train, y_train, learning_rate=1e-3, reg=1e-3, num_iters=50)


    print "Total: ", num_test, '\tRight rate = %f' % (np.mean(y_test == softmax.predict(X_test)), )