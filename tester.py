import cv2
import os
import numpy as np
import faceRecognition as fr

test_img=cv2.imread('TestImages/harsha.jpg')
faces_detected,gray_img=fr.faceDetection(test_img)
print("faces_detected:",faces_detected)

faces,faceID=fr.labels_for_training_data('trainingImages')
face_recognizer=fr.train_classifier(faces,faceID)
face_recognizer.save('trainingData.yml')
#face_recognizer.write('trainingData.yml')

face_recognizer=cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('trainingData.yml')
name={0:"Lohith",1:"Milli",2:"Harsha",3:"Karteshwaran"}


for face in faces_detected:
    (x,y,w,h)=face
    roi_gray=gray_img[y:y+h,x:x+h]
    label,confidence=face_recognizer.predict(roi_gray)#predicting the label of given image
    print("confidence:",confidence)
    print("label:",label)
    fr.draw_rect(test_img,face)
    predicted_name=name[label]
    if(confidence>60):       
       continue
    fr.put_text(test_img,predicted_name,x,y)

resized_img=cv2.resize(test_img,(1000,1000))
cv2.imshow("face detecetion",resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows

