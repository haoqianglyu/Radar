# -*- coding: utf-8 -*-
"""
Created on Thu April 12 15:28:29 2018

@author: lvhaoqiang
"""
from skimage import io,transform
import glob
import os
import tensorflow as tf
import numpy as np

#测试数据集的地址
path='./radar_4/'
#建立的模型存放位置
model_path='./radar_net/model.ckpt'
#图片的格式resize（长宽厚）
#长和宽都为100像素，通道是3通道
width = 100
height = 100
rgb = 3

#读取图片方法
def read_img(path):
    #列表清单文件的统计,x为图片的名称
    cate=[path+x for x in os.listdir(path) if os.path.isdir(path+x)]
#    print(cate)
    imgs=[]
    labels=[]
    for idx,folder in enumerate(cate):
        #glob匹配所有的jpg格式文件
        for im in glob.glob(folder+'/*.png'):
            print('reading the images:'+str(im))
            #通过IO读取合格的image
            img=io.imread(im)
            img=transform.resize(img,(width,height),mode="constant")
            #将读取的img放入到列表中
            imgs.append(img)
            #将每个图片所对应的标签存入列表中
            labels.append(idx)
    #将列表转化成float32的元组
    return np.asarray(imgs,np.float32),np.asarray(labels,np.int32)
#将图片元组和图片标签传入data和label中
data,label=read_img(path)
#一共有800张图片
num_example=data.shape[0]
#乱序
arr=np.arange(num_example)
np.random.shuffle(arr)
data=data[arr]
label=label[arr]

#将图片集合和标签集合传输到训练图片和训练标签中
x_train=data
y_train=label
#print(x_train,y_train)
#每个批次的大小
batch_size = 80
#初始化权值
def weight_variable(name,shape):
#    initial = tf.truncated_normal(shape,stddev=0.1)#生成一个截断的正态分布
    conv1_weights = tf.get_variable(name,shape,initializer=tf.truncated_normal_initializer(stddev=0.1))
    return conv1_weights
#初始化偏置
def bias_variable(name,shape):
    conv1_biases = tf.get_variable(name, shape, initializer=tf.constant_initializer(0.0))
    return conv1_biases
#卷积层
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')
#池化层
def max_pool_2x2(x):
    #ksize [1,x,y,1]
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

#定义一个函数，按批次取数据
def get_nextbatch(inputs=None, targets=None, batch_size=None, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]
#x输入，y输出,100*100*3
#命名空间

x = tf.placeholder(tf.float32,shape = [None,width,height,rgb],name='x')
y = tf.placeholder(tf.int32,shape = [None,],name='y')
x_image = tf.reshape(x,[-1,100,100,3],name='x_image')
        
with tf.variable_scope('layer1-conv1'):
    #初始化第一个卷积层的权值和偏置
    W_conv1 = weight_variable("weight",[5,5,3,32])
    b_conv1 = bias_variable("bias",[32])
    #把x_image和权值向量进行卷积，再加上偏置值，然后应用于relu激活函数
    conv2d_1 = conv2d(x_image,W_conv1) + b_conv1
    h_conv1 = tf.nn.relu(conv2d_1)
    with tf.name_scope('h_pool1'):
        h_pool1 = max_pool_2x2(h_conv1)
with tf.variable_scope("layer2-conv2"):
    #初始化第二个卷积层的权值和偏置
    W_conv2 = weight_variable("weight",[5,5,32,64])
    b_conv2 = bias_variable("bias",[64])
    #把h_pool1和权值向量进行卷积，再加上偏置值，然后应用于relu激活函数
    conv2d_2 = conv2d(h_pool1,W_conv2) + b_conv2
    h_conv2 = tf.nn.relu(conv2d_2)
    with tf.name_scope('h_pool2'):
        h_pool2 = max_pool_2x2(h_conv2)
#卷积完成后开始全连接
with tf.variable_scope('layer3-fc1'):
    #初始化第一个全连接层的权值
    W_fc1 = weight_variable("weight",[25*25*64,1024])
    b_fc1 = bias_variable("bias",[1024])
    #把池化层2的输出扁平化为1维
    h_pool2_flat = tf.reshape(h_pool2,[-1,25*25*64],name='h_pool2_flat')
     #求第一个全连接层的输出
    wx_plus_b1 = tf.matmul(h_pool2_flat,W_fc1) + b_fc1
    h_fc1 = tf.nn.relu(wx_plus_b1)
    h_fc1_drop = tf.nn.dropout(h_fc1,0.7,name='h_fc1_drop')
with tf.variable_scope('layer4-fc2'):
    #初始化第二个全连接层
    W_fc2 = weight_variable("weight",[1024,4])
    b_fc2 = bias_variable("bias",[4])
    prediction = tf.matmul(h_fc1_drop,W_fc2) + b_fc2
b = tf.constant(value=1,dtype=tf.float32)
prediction_eval = tf.multiply(prediction,b,name='prediction_eval') 
#softmax输出概率
#prediction = tf.nn.softmax(wx_plus_b2)
#利用交叉熵来计算损失函数,不可用softmax做为输入
#交叉熵代价函数
cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction, labels=y)

#使用AdamOptimizer进行优化
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
#求准确率
correct_prediction = tf.equal(tf.cast(tf.argmax(prediction,1),tf.int32), y)#argmax返回一维张量中最大的值所在的位置
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        
saver=tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(10):
#十次全训练
        accuracy_train , n_batch= 0,0
        for x_train_a, y_train_a in get_nextbatch(x_train, y_train, batch_size, shuffle=True):
            _,acc = sess.run([train_step,accuracy],feed_dict={x:x_train_a, y:y_train_a})
            accuracy_train += acc
            n_batch += 1
            print("批量准确度:" + str(acc))
        print("------------------------------------------>")
        print ("训练次数:" + str(i+1),"准确度:" + str(accuracy_train/n_batch))
    saver.save(sess,model_path)
    sess.close()