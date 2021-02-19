
import numpy as np 
import cv2 
   
      

feature_img = cv2.imread('/home/karel/Downloads/green.jpg')    
feature_img_bw = cv2.cvtColor(feature_img,cv2.COLOR_BGR2GRAY) 
orb = cv2.ORB_create() 
queryKeypoints, queryDescriptors = orb.detectAndCompute(feature_img_bw,None) 
feature_img = cv2.drawKeypoints(feature_img, queryKeypoints, outImage = None, color=(0, 255, 0), flags=0)  
feature_img = cv2.resize(feature_img, (700,700)) 
cv2.imshow("Matches", feature_img) 
cv2.waitKey(3000) 
