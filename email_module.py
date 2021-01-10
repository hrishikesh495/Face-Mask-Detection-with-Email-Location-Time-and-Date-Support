# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 02:25:58 2021

@author: Hrishikesh Sunil Shinde
@Programming Language: Python
@IDE: Spyder
@Platform: Windows 10

"""
#%% Get Current GeoLocation
from requests import get
import urllib
import urllib.request
import ast


def currentLocation():
    ip = get('https://api.ipify.org').text
    url = 'http://ip-api.com/json/' + ip
    req = urllib.request.Request(url)
    out = urllib.request.urlopen(req).read()
    
    dict_str = out.decode("UTF-8")
    mydata = ast.literal_eval(dict_str)
    loc = str(mydata['city'])+", "+str(mydata['regionName'])+", "+str(mydata['country'])+"\n"+"[Latitude, Longitude]: ["+str(mydata['lat'])+", "+str(mydata['lon'])+"]\n"+"TimeZone: "+str(mydata['timezone'])+"\n"+"IP: "+str(mydata['query'])+"\n"
    return loc


# Return Current System Date and Time
import datetime

def date_time():
    current_time = datetime.datetime.now() 
    loc=currentLocation()
    
    Date = "Date: "+str(current_time.day)+"/"+str(current_time.month)+"/"+str(current_time.year)
    Time = "Time: "+str(current_time.hour)+":"+str(current_time.minute)+":"+str(current_time.second)
    detect = "Person Without Mask Detected!\nCamera Number: 1\nActivity Detected On: "
    location = "Location: "+loc
    new_line ="\n"
    activity = detect+new_line+new_line+str(Date)+new_line+str(Time)+new_line+new_line+location+new_line
    return activity



#%% Sends Email with attachments
import smtplib 
from email.mime.multipart import MIMEMultipart 
from email.mime.text import MIMEText 
from email.mime.base import MIMEBase 
from email import encoders 
from multiprocessing import Lock



def send_email():
    
    fromaddr = "mask.surveillance@gmail.com"
    # creates SMTP session
    s = smtplib.SMTP('smtp.gmail.com', 587) 

    # start TLS for security 
    s.starttls() 

    # Authentication 
    s.login(fromaddr, "surveillanceIndia") 
    print("\n\n\nEmail Login Success\n\n\n\n\n")
    #lock.acquire()
    
    fromaddr = "mask.surveillance@gmail.com"
    toaddr = "surveillance.team.india@gmail.com"

    # instance of MIMEMultipart 
    msg = MIMEMultipart() 

    # storing the senders email address   
    msg['From'] = fromaddr 

    # storing the receivers email address  
    msg['To'] = toaddr 

    # storing the subject  
    msg['Subject'] = "Camera 1 - Detect Person without Mask"

    # string to store the body of the mail 

    activity = date_time()
    body = str(activity)

    # attach the body with the msg instance 
    msg.attach(MIMEText(body, 'plain')) 

    # open the file to be sent  
    filename1 = "Captured Snapshot 1.png"
    filename2 = "Captured Snapshot 2.png"
    filename3 = "Captured Snapshot 3.png"
    
    attachment1 = open(r"Detected/image 1.png", "rb") 
    attachment2 = open(r"Detected/image 2.png", "rb") 
    attachment3 = open(r"Detected/image 0.png", "rb") 

    # instance of MIMEBase and named as p 
    p1 = MIMEBase('application', 'octet-stream')
    p2 = MIMEBase('application', 'octet-stream') 
    p3 = MIMEBase('application', 'octet-stream') 

    # To change the payload into encoded form 
    p1.set_payload((attachment1).read())
    p2.set_payload((attachment2).read())
    p3.set_payload((attachment3).read())


    # encode into base64 
    encoders.encode_base64(p1) 
    encoders.encode_base64(p2)
    encoders.encode_base64(p3) 

    p1.add_header('Content-Disposition', "attachment; filename= %s" % filename1)
    p2.add_header('Content-Disposition', "attachment; filename= %s" % filename2)
    p3.add_header('Content-Disposition', "attachment; filename= %s" % filename3)

    # attach the instance 'p' to instance 'msg' 
    msg.attach(p1)
    msg.attach(p2)
    msg.attach(p3)



    # Converts the Multipart msg into a string 
    text = msg.as_string() 

    # sending the mail 
    s.sendmail(fromaddr, toaddr, text) 

    # terminating the session 
    #lock.release()
    print("\n\n\n\nMail Sent Success\n\n\n\n")
    

    