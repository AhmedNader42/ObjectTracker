from centroid_tracker import CentroidTracker
import numpy as np
import imutils
import time
import cv2
import os.path




ct = CentroidTracker()
(H, W) = (None, None)
model = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000_fp16.caffemodel")
vc = cv2.VideoCapture(0)
time.sleep(2.0)


isAvailable, frame = vc.read()
count = 0
while True and isAvailable:
    isAvailable, frame = vc.read()
    frame = imutils.resize(frame, width=400)

    # cv2.imshow("d", frame[40:320, 132:390])
    if W is None or H is None:
        (H, W) = frame.shape[:2]
        
        
    blob = cv2.dnn.blobFromImage(frame, 1.0, (W, H), (104.0, 177.0, 123.0))
    model.setInput(blob)
    detections = model.forward()
    rects = []
    
    for i in range(0, detections.shape[2]):
        if detections[0, 0, i, 2] > 0.5:
            
            box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
            rects.append(box.astype("int"))
            
            (startX, startY, endX, endY) = box.astype("int")
            coordinates = box.astype("int")
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

            

    objects = ct.update(rects)
    
    for (objectID, centroid) in objects.items():
        if not os.path.isfile("{}.jpg".format(objectID)):
            img = frame[coordinates[1]:coordinates[0], coordinates[3]:coordinates[2]]
            cv2.imwrite("{}.jpg".format(objectID), img)
            cv2.imshow("{} img".format(objectID), img)
        text = "Face {}".format(objectID)
        cv2.putText(frame, text, (centroid[0]-10 , centroid[1]-10), cv2.FONT_ITALIC, 0.5, (0, 0, 255), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 5, (0, 0, 255), -1)

        

        
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord("q") or not isAvailable:
        break

vc.release()        
cv2.destroyAllWindows()
