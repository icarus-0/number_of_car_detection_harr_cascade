#!/usr/bin/env python
# coding: utf-8

# In[2]:



import cv2
print(cv2.__version__)

cascade_src = 'D:\\datasets\\opencv-master\\data\\haarcascades\\cars.xml'
#video_src = 'C:\\Users\\Naman\\Downloads\\Compressed\\vehicle_detection_haarcascades-master\\vehicle_detection_haarcascades-master\\dataset\\video1.avi'
#video_src = "D:\\datasets\\example1.mp4"
video_src = 'C:\\Users\\Naman\\Downloads\\Compressed\\vehicle_detection_haarcascades-master\\vehicle_detection_haarcascades-master\\dataset\\video2.avi'
font = cv2.FONT_HERSHEY_SIMPLEX
cap = cv2.VideoCapture(video_src)
car_cascade = cv2.CascadeClassifier(cascade_src)
total_cars = 0
flag = 19
while True:
    flag += 1
    ret, img = cap.read()
    if (type(img) == type(None)):
        break
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #img=cv2.resize(img,(400,400))
    cars = car_cascade.detectMultiScale(gray, 1.1, 1)

    for (x,y,w,h) in cars:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
    
    if flag%20 ==0:
        total_cars += len(cars)
    cv2.putText(img,str(total_cars),(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
    
    cv2.imshow('video', img)
    
    if cv2.waitKey(33) == 27:
        break

cv2.destroyAllWindows()


# In[ ]:





# In[ ]:




