#%%
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import pandas as pd 
import glob
import os 
import random
import pickle
#%%

DATADIR = "E:\Tanay_projects\Dogs_cats\kagglecatsanddogs_3367a\PetImages"

CATEGORIES = ["Dog", "Cat"]

training_data = []
IMG_SIZE= 100 

def create_training_data():
    for category in CATEGORIES:

        path = os.path.join(DATADIR,category)  
        class_num = CATEGORIES.index(category)  

        for img in tqdm(os.listdir(path)):  
            try:
                img_array = cv2.imread(os.path.join(path,img))  
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  
                training_data.append([new_array, class_num])  
            except Exception as e:  
                pass

create_training_data()
#%%
random.shuffle(training_data)

print(training_data[:10])
#%%
x = []
y = []
for features,label in training_data:
    x.append(features)
    y.append(label)

X = np.array(x).reshape(-1,IMG_SIZE,IMG_SIZE,3)

pickle_out = open("X.pickle",'wb')
pickle.dump(X,pickle_out)

pickle_out = open("y.pickle",'wb')
pickle.dump(y,pickle_out)
#%%
import pickle

pickle_in = open('E://Tanay_projects//Dogs_cats//X.pickle','rb')
X = pickle.load(pickle_in)

pickle_in = open('E://Tanay_projects//Dogs_cats//y.pickle','rb')
y = pickle.load(pickle_in)

#Normalizing the Data
x = X/255.0

#print(x[1])
#%%
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,Dense,Flatten,Dropout,Activation,MaxPooling2D

 #create Model 
model = Sequential()

model.add(Conv2D(128,(3,3),input_shape=x.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(128,(3,3),input_shape=x.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(256,(3,3),input_shape=x.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Flatten())

model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(x, y, batch_size=128,epochs=15, validation_split=0.1)


model.save('dogs_cats_cnn.model')
#%%
#predication=model.predict(x)
'''import cv2
new_img = 'E://Tanay_projects//Dogs_cats//kagglecatsanddogs_3367a//PetImages//Cat//533.jpg'

img_array = cv2.imread(new_img)  
new_array = cv2.resize(img_array, (100,100)) 

new=np.array(new_array).reshape(-1,100,100,3)
new=new/255.0

pred= model.predict(new)

print(pred)'''





#%%


#%%

