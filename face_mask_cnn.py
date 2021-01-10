# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 01:15:03 2021

@author: Hrishikesh Sunil Shinde
@Programming Language: Python
@IDE: Spyder
@Platform: Windows 10

"""
#%% 1) Importing Libraries for training CNN

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten,Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import tensorflow as tf
gpus= tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


#%% 2) Importing Dataset

# a) Creating Image Data Generators used for data augmentation

train_datagen = ImageDataGenerator(brightness_range=[0.1,1.0],rescale = 1./255,horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255,horizontal_flip = True,brightness_range=[0.3,1.0])

#%% Importing and Augmenting Training Images

""" Binary Classification

Class 0: WithMask
CLass 1: Without Mask"""

# Training Images: 10052 of 2 classes
total_training_imgs = 10052

# Validation Images: 800 of 2 classes
total_validation_imgs = 800

# Test Images: 992 of 2 classes
total_test_imgs = 992

train_path = r'Face Mask Dataset\Train'
validation_path = r'Face Mask Dataset\Validation'
test_path = r'Face Mask Dataset\Test'

# a) Importing and Augmenting Training Images
train_it = train_datagen.flow_from_directory(train_path, class_mode='binary', batch_size=16,shuffle = True,target_size = (64,64))

# b) Importing and Augmenting Validation Images
validate_it = test_datagen.flow_from_directory(validation_path, class_mode='binary', batch_size=16,shuffle = False,target_size = (64,64))

# c) Importing and Augmenting Test Images
test_set = test_datagen.flow_from_directory(test_path, class_mode='binary', batch_size=16,shuffle = False,target_size = (64,64))

#%% 3. Building CNN Model

# a) Creating a sequential Model
model = Sequential()

# b) Adding Hidden Layers

# Hidden Layer 1: Conv + Activation + Pooling
model.add(Conv2D(32,(3,3),padding='same',input_shape=(64, 64, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Hidden Layer 2: Conv + Activation + Pooling
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Hidden Layer 3: Conv + Activation + Pooling
model.add(Conv2D(64, (3, 3),padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Hidden Layer 4: Conv + Activation + Dropout
model.add(Conv2D(128, (3, 3),padding = 'same'))
model.add(Activation('relu'))
model.add(Dropout(0.25))

# Hidden Layer 5: Conv + Activation + Dropout
model.add(Conv2D(128, (3, 3),padding = 'same'))
model.add(Activation('relu'))
model.add(Dropout(0.5))

# c) Adding Fully Connected Layer
model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(1))
model.add(Activation('sigmoid'))

#%% 4. Compiling the created model

model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
#%% Model Summary

print(model.summary())

#%% 5. Callbacks (Checkpoints,earlystopping & learning rate)

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# a) Create Checkpoints to save best model 

checkpoint = ModelCheckpoint(r"Saved_Model\Face_Mask_Best_Model",monitor="val_loss",mode="min",save_best_only=True,verbose= 1)

# b) If validation accuracy not increasing stop early (Early Stopping)
earlyStop = EarlyStopping(monitor = "val_loss",min_delta = 0,patience =5,verbose =1,restore_best_weights=True)

# c) If valdation accuracy not increasing reduce  Learning Rate
reduce_lr = ReduceLROnPlateau(monitor="val_loss",factor =0.2,patience=3,verbose =1,min_delta=0.0001)

# callbacks
callbacks=[earlyStop,reduce_lr,checkpoint]

#%% 6. Training the CNN

# Fitting the model on training dataset
epoch = 50

history = model.fit(train_it,steps_per_epoch = total_training_imgs // 16,validation_data=validate_it,validation_steps=total_validation_imgs // 16,epochs=epoch,callbacks=callbacks)

