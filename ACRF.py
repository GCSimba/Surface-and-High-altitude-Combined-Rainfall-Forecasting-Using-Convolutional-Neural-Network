#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from numpy.random import RandomState
import pandas as pd
import numpy as np
import time


# In[5]:


#定义神经网络的参数
d=30#输入节点个数
l=1#输出节点个数
q=2*d+1#隐层个数,采用经验公式2d+1
eta=0.5#学习率
error=0.002#精度
 
#初始化权值和阈值
w1= tf.Variable(tf.random_normal([d, q], stddev=1, seed=1))#seed设定随机种子，保证每次初始化相同数据
b1=tf.Variable(tf.constant(0.0,shape=[q]))
w2= tf.Variable(tf.random_normal([q, l], stddev=1, seed=1))
b2=tf.Variable(tf.constant(0.0,shape=[l]))
 
#输入占位
x = tf.placeholder(tf.float32, shape=(None, d))
y_= tf.placeholder(tf.float32, shape=(None, l))
 
#构建图：前向传播
a=tf.nn.sigmoid(tf.matmul(x,w1)+b1)#sigmoid激活函数
y=tf.nn.sigmoid(tf.matmul(a,w2)+b2)
mse = tf.reduce_mean(tf.square(y_ -  y))#损失函数采用均方误差
train_step = tf.train.AdamOptimizer(eta).minimize(mse)#Adam算法


# In[6]:


def noraml(data):
    ''''''
    ds = data.shape
    aa = data.flatten()
    max_a = max(aa)
    min_a = min(aa)
    newdata = np.array([round((ai-min_a)/(max_a-min_a),4) for ai in aa])
    return newdata.reshape(ds), max_a-min_a,min_a


x1_train, x1_test, x2_train, x2_test, y_train, y_test  = [], [], [], [], [], []
with open('new_train2.dat','r') as f0:
    for l in f0:
        ll = l.split()
        aa = []
        for i in range(3,47,4): aa.extend([float(ll[i]),float(ll[i+1])])
        bb = [round(float(j),4) for j in ll[49:-1]]
        y_train.append(float(ll[48]))
        x1_train.append(aa+bb)
with open('new_test.dat','r') as f0:
    for l in f0:
        ll = l.split()
        aa = []
        for i in range(3,47,4): aa.extend([float(ll[i]),float(ll[i+1])])
        bb = [round(float(j),4) for j in ll[49:-1]]
        y_test.append(float(ll[48]))
        x2_test.append(aa+bb)
        
x_train = np.array(x1_train)
x_test = np.array(x2_test)
print (len(x_train),len(x_test))

y_train = np.array(y_train)
y_test = np.array(y_test)

X, dif, a = noraml(x_train)
testX, destest, b = noraml(x_test)
Y, die, c = noraml(y_train)
testY, des, nummin= noraml(y_test)

Y = Y.reshape(len(y_train), 1)
testY = testY.reshape(len(y_test), 1)


# In[7]:


start = time.clock()
#创建会话来执行图
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()#初始化节点
    sess.run(init_op)

    STEPS=0
    while True:
        sess.run(train_step, feed_dict={x: X, y_: Y})
        STEPS+=1
        train_mse= sess.run(mse, feed_dict={x: X, y_: Y})
        print (train_mse)
        if STEPS % 5 == 0:#每训练100次，输出损失函数
            print("第 %d 次训练后,训练集损失函数为：%g" % (STEPS, train_mse))
        if train_mse<error:
            break
    print("总训练次数：",STEPS)
    end = time.clock()
    print("运行耗时(s)：",end-start)
    Normal_y= sess.run(y, feed_dict={x: testX})#求得测试集下的y计算值
#     test_mse= sess.run(mse, feed_dict={y: Normal_y, y_: testY})
    DeNormal_y=Normal_y*dif+nummin#将y反归一化
    test_mse= sess.run(mse, feed_dict={y: DeNormal_y, y_: testY})#计算均方误差
    print("测试集均方误差为：",test_mse)
    print (DeNormal_y)
    with open('bp_res','w') as f0:
        for i,de in enumerate(DeNormal_y):
            d = de[0]
            if de[0] <= 0.000036: d = 0
            f0.writelines('{}\t{}\t{}\n'.format(d,y_test[i],testY[i][0]))



