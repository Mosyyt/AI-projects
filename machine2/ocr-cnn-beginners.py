#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import cv2
import numpy as np


images = []
labels = []

path = '/kaggle/input/standard-ocr-dataset/data/training_data'

dir_list = os.listdir(path)
for i in dir_list:
  dir = os.path.join(path, i)
  file_list = os.listdir(dir)
  for j in file_list:
    files = os.path.join(dir, j)
    img = cv2.imread(files)
    img = cv2.resize(img, (64,64))
    img = np.array(img, dtype=np.float32)
    img = img/255
    images.append(img)
    labels.append(i)


# In[2]:


X = np.array(images)
len(X)


# In[3]:


X.shape


# In[4]:


y = np.array(labels)
len(y)


# In[5]:


y.shape


# In[6]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)


# In[7]:


from sklearn.utils import shuffle
X_sh, y_sh = shuffle(X, y, random_state=42)


# In[8]:


from keras.models import Sequential
from keras.layers import Conv2D, Flatten, MaxPooling2D, Dense


# In[9]:


model = Sequential()


# In[10]:


model.add(Conv2D(filters=16, kernel_size=(3,3), activation='relu', input_shape=(64,64,3)))
model.add(MaxPooling2D())
model.add(Conv2D(filters=32, kernel_size=(3,3),  activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(filters=64, kernel_size=(3,3),  activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=36, activation='softmax'))


# In[11]:


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics = ['accuracy'])


# In[12]:


history = model.fit(X_sh, y_sh ,validation_split=0.2, batch_size=25, epochs=10)


# In[13]:


import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['loss', 'val_loss'])


# In[14]:


test_images = []
test_labels = []

path = '/kaggle/input/standard-ocr-dataset/data/testing_data'

dir_list = os.listdir(path)
for i in dir_list:
  dir = os.path.join(path, i)
  file_list = os.listdir(dir)
  for j in file_list:
    files = os.path.join(dir, j)
    img = cv2.imread(files)
    img = cv2.resize(img, (64,64))
    img = np.array(img, dtype=np.float32)
    img = img/255
    test_images.append(img)
    test_labels.append(i)


# In[15]:


X_test = np.array(test_images)
y_test = np.array(test_labels)


# In[16]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_test = le.fit_transform(y_test)


# In[17]:


test_loss, test_accuracy = model.evaluate(X_test, y_test)


# In[18]:


print(test_loss,test_accuracy )


# In[ ]:




