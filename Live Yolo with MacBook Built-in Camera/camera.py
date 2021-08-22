import numpy as np
import cv2
from yolo import predict
from PIL import Image
cap = cv2.VideoCapture(0)
cv2.waitKey(1000)
print("Start.") 
i=0
key=1
while(i<100):
        ret, frame = cap.read()
        if cv2.waitKey(200)  == 27:
                break
        i+=1
        frame=predict(frame)
        cv2.imshow("PIC",frame)
        #cv2.imwrite(f"/Users/Dennis/Desktop/hand genture project/pics/image-{i}.png",frame)
cap.release()
