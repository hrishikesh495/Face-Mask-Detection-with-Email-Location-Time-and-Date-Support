# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 02:33:55 2021

@author: Hrishikesh Sunil Shinde
@Programming Language: Python
@IDE: Spyder
@Platform: Windows 10

"""
#%% Memory Growth
import tensorflow as tf
gpus= tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

#%% Importing Libraries
import tensorflow 
from tensorflow.keras.preprocessing.image import img_to_array
import cv2
from cv2 import resize
from multiprocessing import Process
import email_module
import numpy as np

#%% Loading Trained CNN Model

# Loading our CNN
model = tensorflow.keras.models.load_model('Saved_Model\Face_Mask_Best_Model')

# Loading Classifier to detect face
face_classifier = cv2.CascadeClassifier(r'Classifier/haarcascade_frontalface_default.xml')

#%% Face Mask Detection using OpenCV

# Global Variables
pred = ""
photo_capture_counter=0
counter=1

# mask detection function

#img is BGR image & img2 is RGB image
def face_mask_detection(img,img2):
    
    global counter,lock,photo_capture_counter
    
    #converting to grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    # passing the grey scale image to classifier to detecte face from the frame
    faces = face_classifier.detectMultiScale(gray,1.3,5)
    
    #if face not detected in the frame    
    if faces == ():
        
        #flip the RGB image and return the image
        img2 = cv2.flip(img2,1)
        # put No face detected on the frame
        cv2.putText(img2, "No Face Found", (240,320) , cv2.FONT_HERSHEY_SIMPLEX,1, (255,0,0), 2)
        counter = 0 #damage counter reset
        photo_capture_counter = 0 # file write counter reset
        return img2
        
    
    #if face is detected 
    for(x,y,w,h) in faces:
        
        # Crop the face from the whole image
        croppedImage = img[y:y+h, x:x+w].copy()
        
        #resize the cropped image to 64*64 (i.e CNN input layer take 64*64)
        try:
            croppedImage = resize(croppedImage,(64,64), interpolation = cv2.INTER_AREA)
        except:
            return img2
        
        # convert the image into array
        croppedImage = img_to_array(croppedImage)
        croppedImage = croppedImage.reshape(1, 64, 64, 3)
        croppedImage = croppedImage.astype('float32')
        
        # predict the face has or not 
        result = model.predict(croppedImage)

        # Class Indices 0: With Mask,  1:  Without Mask      
        if result[0][0]==0:
            pred = "With Mask"
        else:
            pred = "Without Mask"
    
    # Adds green color rectangle if using mask 
    # Adds red color rectangle if not using mask
    if(pred == "With Mask"):
        cv2.rectangle(img2,(x-25,y-25),(x+w+25,y+h+25),(0,255,0),2)
    else:
        cv2.rectangle(img2,(x-25,y-25),(x+w+25,y+h+25),(0,0,255),2)
        
    #flip the image
    img2 = cv2.flip(img2,1)
      
    # label location
    text_loc_x = 480-x
    text_loc_y = y+h+53
    
    # Add Label With Mask/ Without Mask on the frame
    if(pred == "With Mask"):
        box_x,box_y,box_w,box_h = text_loc_x,text_loc_y-28,180,40
        cv2.rectangle(img2, (box_x, box_y), (box_x + box_w, box_y + box_h), (0,255,0), -1)
        box_x,box_y,box_w,box_h = text_loc_x-7,y-65,80,40
        cv2.rectangle(img2, (box_x, box_y), (box_x + box_w, box_y + box_h), (0,255,0), -1)
        
        cv2.putText(img2, " "+ pred, (text_loc_x, text_loc_y) , cv2.FONT_HERSHEY_SIMPLEX,1, (255,255,255), 2)
        cv2.putText(img2, "Safe", (480-x, y-30) , cv2.FONT_HERSHEY_SIMPLEX,1, (255,255,255), 2)
        counter = 1 #reset counter to 1
        photo_capture_counter = 0 # reset write file counter to 0
    else:
        box_x,box_y,box_w,box_h = text_loc_x-35,text_loc_y-29,220,40
        cv2.rectangle(img2, (box_x, box_y), (box_x + box_w, box_y + box_h), (0,0,255), -1)
        cv2.putText(img2, " "+ pred, (text_loc_x-45, text_loc_y) , cv2.FONT_HERSHEY_SIMPLEX,1, (255,255,255), 2)
        ans = counter # To display danger meter
        
        if(counter>98):
           counter = 1 #reset damage counter
         
        # adds frame captured text on the frame
        if (counter >20 and counter<30) or (counter >50 and counter<60) or (counter>80 and counter<90):
            box_x,box_y,box_w,box_h = 0,0,300,50
            cv2.rectangle(img2, (box_x, box_x), (box_x + box_w, box_y + box_h), (0,0,0), -1)
            cv2.putText(img2, "Frame Captured ", (15, 30) , cv2.FONT_HERSHEY_SIMPLEX,1, (255,0,255), 2)

        # 3 frames are written on to the local system at damage counter 30,60,90
        if counter%30==0:
            box_x,box_y,box_w,box_h = 480-x-120,y-67,280,40
            cv2.rectangle(img2, (box_x, box_y), (box_x + box_w, box_y + box_h), (0,255,255), -1)
            cv2.putText(img2, "Danger meter: "+str(ans), (480-x-120, y-33) , cv2.FONT_HERSHEY_SIMPLEX,1, (255,0,255), 2)
            cv2.imwrite('Detected\image '+str(photo_capture_counter)+'.png',img2)
            print("\n\n\nFrame Captured "+str(photo_capture_counter+1)+"\n\n\n")
               
            photo_capture_counter+=1 # increment file write counter
            counter +=1 # increment damage counter

            if photo_capture_counter==3:
                print(counter)
                photo_capture_counter=0
                p1=Process(target=email_module.send_email)

                p1.start()
                #p1.join() #if on then main programs stops execution till process p1 gets complete
            
                print("\n\n\nEmail To Survellience Team Sent\n\n\n")
        else:
            # displays text in yellow color
            if counter<70:
                box_x,box_y,box_w,box_h = 480-x-120,y-67,280,40
                cv2.rectangle(img2, (box_x, box_y), (box_x + box_w, box_y + box_h), (0,255,255), -1)
                cv2.putText(img2, "Danger meter: "+str(ans), (480-x-120, y-33) , cv2.FONT_HERSHEY_SIMPLEX,1, (255,0,255), 2)
            # displays text in red color
            else:  
                box_x,box_y,box_w,box_h = 480-x-120,y-67,280,40
                cv2.rectangle(img2, (box_x, box_y), (box_x + box_w, box_y + box_h), (0,0,255), -1)
                cv2.putText(img2, "Danger meter: "+str(ans), (480-x-120, y-33) , cv2.FONT_HERSHEY_SIMPLEX,1, (0,255,255), 2)

            counter +=1 #incrementing damage counter
            
        # Display Email Sending Frame
        if(counter>86 and counter <91):
            blank_image = np.zeros((480,640,3), np.uint8)
            img2 = blank_image
            cv2.putText(img2, "Sending 3 Snaps via Email To Survellience Team ", (30, 250) , cv2.FONT_HERSHEY_SIMPLEX,0.5, (127,255,127), 1)
            cv2.putText(img2, "Please Wait!! ", (50, 300) , cv2.FONT_HERSHEY_SIMPLEX,0.5, (127,255,127), 1)
        # Display Email Sent Successful frame
        if(counter>=91 and counter <98):
            blank_image = np.zeros((480,640,3), np.uint8)
            img2 = blank_image
            cv2.putText(img2, "Email Sent Successfully ", (30, 250) , cv2.FONT_HERSHEY_SIMPLEX,1.6, (127,255,127), 1)

    return img2

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    
    while(True):
        # Gets the frame from camera
        ret,frame = cap.read()
        #cloning the frame
        temp = frame
        # BGR to RGB (Open CV: BGR, Keras: RGB)
        frame = frame[:,:,::-1]
        
        #Displaying the Result
        cv2.imshow('Live Face Detection',face_mask_detection(frame,temp))
        if cv2.waitKey(1)==13: # Exit when 'Enter' button is Pressed
            break
    photo_capture_counter=0
    cap.release()
    cv2.destroyAllWindows() #Close All Windows
    
