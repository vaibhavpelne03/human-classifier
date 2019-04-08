#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


# In[2]:


#initialiseing the CNN
classifier = Sequential()


# In[ ]:


#if image size requird then use this parameter 
# img_width = 256
# img_height = 256


# In[3]:


classifier.add(Conv2D( 32,(3,3), input_shape=( 64, 64, 3 ), activation = 'relu'))

#when image size give replce input_shape with img_width and img_height
#classifier.add(Conv2D( 32,(3,3), input_shape=( img_width, img_height, 3 ), activation = 'relu'))

#32  filter (feature detector) then 64 then 128 like that , 3 * 3 fetaure detector
#input shape - forcet to fixed image -3D ALL colored 256,256, 
#because we useing cpu s0 64,64,3 


# In[4]:


classifier.add(MaxPooling2D(pool_size = (2,2)))


# In[5]:


# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))


# In[6]:


classifier.add(Flatten())


# In[7]:


classifier.add(Dense(units = 128, activation='relu'))
classifier.add(Dense(units = 1, activation='sigmoid'))


# In[8]:


classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# In[9]:


#https://keras.io/preprocessing/image/
from keras.preprocessing.image import  ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)


test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        'humandata/humantrain',
        #when the image perticular size given then use
        #target_size=(img_width, img_height),
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        'humandata/humanvalidation',
        #when the image perticular size given then use 
        #target_size=(img_width, img_height),
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(train_generator,
        #std steps_per_erpoch = 8000
        steps_per_epoch=2000,
        #std epochs = 25
        epochs=10,
        validation_data=validation_generator,
        #std validation steps 2000
        validation_steps=800)


# In[36]:


import numpy as np
from keras.preprocessing import image
test_image=image.load_img("humandata/testimg/applauding_028.jpg",target_size=(64,64))
test_image=image.img_to_array(test_image)
test_image=np.expand_dims(test_image,axis=0)
result=classifier.predict(test_image)
trainig_set=classifier.predict(test_image)
train_generator.class_indices
if result[0][0]==1:
    prediction='Not Human'
    print("Not Human")
else:
    prediction='Human'
    print("Human")


# In[38]:


import numpy as np
from keras.preprocessing import image
test_image=image.load_img("humandata/testimg/69020950.jpg",target_size=(64,64))
test_image=image.img_to_array(test_image)
test_image=np.expand_dims(test_image,axis=0)
result=classifier.predict(test_image)
trainig_set=classifier.predict(test_image)
train_generator.class_indices
if result[0][0]==1:
    prediction='Not Human'
    print("Not Human")
else:
    prediction='Human'
    print("Human")

