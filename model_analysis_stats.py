# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 02:02:03 2021

@author: Hrishikesh Sunil Shinde
@Programming Language: Python
@IDE: Spyder
@Platform: Windows 10

"""

#%% 1. Load out trained CNN Model 

from tensorflow import keras

model = keras.models.load_model('Saved_Model\Face_Mask_Best_Model')


#%% Prediction on test set
from tensorflow.keras.preprocessing.image import ImageDataGenerator

testing_datagen = ImageDataGenerator()
test_set = testing_datagen.flow_from_directory(r'Face Mask Dataset\Test', class_mode='binary', batch_size=16,shuffle = False,target_size = (64,64))
preds = model.predict(test_set,verbose=1)

#%% Ploting Graph to check the correct results
import numpy as np
import matplotlib.pyplot as plt
x=np.array(range(0, len(preds)))

plt.scatter(x,preds)

#%% Class Indices

print(test_set.class_indices)

#%% Learning Rate Curve

from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt
fpr, tpr, _ = roc_curve(test_set.classes, preds)
roc_auc = auc(fpr, tpr)

lw = 2
plt.plot(fpr, tpr, color='darkorange',lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

#%% Prediction and Display of 10 random Images from test set using OpenCV

import cv2
import random
from tensorflow.keras.preprocessing import image

with_mask = "Face Mask Dataset\\Test\WithMask\\"
without_mask = "Face Mask Dataset\\Test\WithoutMask\\"
pred = ""
true = ""
for i in range(0,10):
    n = random.randint(0,400)
    if i%2==0:
        final_path = with_mask+"image ("+str(n)+").png"
        true = "With Mask"
    else:
        final_path = without_mask+"image ("+str(n)+").png"
        true = "Without Mask"
    img = image.load_img(final_path, target_size=(64, 64))
    img = image.img_to_array(img)
    img = img.reshape(1, 64, 64, 3)
    #print(np.shape(img))
    img = img.astype('float32')
    result = model.predict(img)
    if result[0][0]==0:
        pred = "With Mask"
    else:
        pred = "Without Mask"
        
    #print(result[0][0])
    BLACK = [0,0,0]
    im = cv2.imread(final_path)
    im = cv2.resize(im, (300,300), fx=5, fy=5, interpolation = cv2.INTER_CUBIC)
    expanded_image = cv2.copyMakeBorder(im, 180, 0, 0, 300 ,cv2.BORDER_CONSTANT,value=BLACK)
    cv2.putText(expanded_image, "Predited - "+ pred, (20, 60) , cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,255), 2)
    cv2.putText(expanded_image, "True - "+true, (20, 120) , cv2.FONT_HERSHEY_SIMPLEX,1, (0,255,0), 2)
    cv2.putText(expanded_image, "Press any Keyboard key for next", (20, 160) , cv2.FONT_HERSHEY_SIMPLEX,1, (255,0,0), 2)
    cv2.imshow("Prediction", expanded_image)
    cv2.waitKey(0)

cv2.destroyAllWindows()
