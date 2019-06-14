# -*- coding: utf-8 -*-
"""
Created on Thu April 12 19:54:20 2018

@author: lvhaoqiang
"""
from skimage import io,transform
import tensorflow as tf
import numpy as np
import os

animal_dict = {0:'norain-nowind',1:'norain-wind',2:'rain-nowind',3:'rain-wind'}

w=100
h=100
c=3

def read_one_image(path):
    img = io.imread(path)
    img = transform.resize(img,(w,h))
    return np.asarray(img)
string = os.listdir("C:/Users/emotion/Desktop/modify_4class/test_ra")
data = []
path = []
for load in string:
    new_load = "C:/Users/emotion/Desktop/modify_4class/test_ra/"+load
    path.append(load)
    l_data = read_one_image(new_load)
    data.append(l_data)
#    print(data)
with tf.Session() as sess:
    saver = tf.train.import_meta_graph('C:/Users/emotion/Desktop/modify_4class/radar_net/model.ckpt.meta')
    saver.restore(sess,tf.train.latest_checkpoint('C:/Users/emotion/Desktop/modify_4class/radar_net'))
    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("x:0")
    feed_dict = {x:data}
   
    logits = graph.get_tensor_by_name("prediction_eval:0")
#    y_soft = graph.get_tensor_by_name("softmax_probability:0")

    classification_result = sess.run(logits,feed_dict)
  
    output = []
    output = tf.argmax(classification_result,1).eval()
    step,norain_nowind,norain_wind,rain_nowind,rain_wind = 0,0,0,0,0
    for i in range(len(output)):
        if animal_dict[output[i]] == "norain-nowind":
            norain_nowind += 1
        elif animal_dict[output[i]] == "norain-wind":
            norain_wind += 1
        elif animal_dict[output[i]] == "rain-nowind":
            rain_nowind += 1
        elif animal_dict[output[i]] == "rain-wind":
            rain_wind += 1
       
        print(path[i]+'的类型是:'+animal_dict[output[i]])
        step += 1
    print(classification_result)
    print("无雨无风率:"+str(norain_nowind/step),"无雨有风率:"+str(norain_wind/step),"有雨无风率:"+str(rain_nowind/step),"有雨有风率:"+str(rain_wind/step))
