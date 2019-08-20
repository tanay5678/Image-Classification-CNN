# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 15:32:10 2019

@author: Tanay
"""

import numpy as np
import cv2
import tensorflow as tf 

category = ['Dog','Cat']

def prepare(filepath):
    img_array = cv2.imread(filepath)  
    new_array = cv2.resize(img_array, (100,100))
    return new_array.reshape(-1,100,100,3)

#load The Model
  
Cnn = tf.keras.models.load_model('E://Tanay_projects//Dogs_cats//dogs_cats_cnn.model')


predication = Cnn.predict([prepare('E://Tanay_projects//Dogs_cats//Test_dog.jpg')])

print(category[int(predication)])