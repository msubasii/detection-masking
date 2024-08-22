import cv2
import numpy as np

#Loading Mask RCNN
net =cv2.dnn.readNetFromTensorflow("dnn/frozen_inference_graph.pb",
                                   "dnn/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt")

#generate random colors
colors = np.random.randint(0,255,(100,3))

#Loading Image
img=cv2.imread("road2.jpg")

#Resizing Image
img=cv2.resize(img,(800,600))
height,width,_=img.shape

#create black image
black_image=np.zeros((height,width,3),np.uint8)
black_image[:]=(100,100,0)

#Bu fonksiyon,görüntüyü derin öğrenme modelleri için uygun bir formata dönüştürür
blob = cv2.dnn.blobFromImage(img, swapRB=True)
net.setInput(blob)


#for each box we have associated mask
boxes, masks = net.forward(["detection_out_final", "detection_masks"])
detection_count = boxes.shape[2]

for i in range(detection_count):
    box = boxes[0, 0, i]
    class_id=box[1]
    score =box[2]
    if score < 0.5:
        continue

    #get box coodinates
    y =int(box[4]*height)
    x =int(box[3]*width)
    x2=int(box[5]*width)
    y2=int(box[6]*height)

    roi=black_image[y:y2,x:x2]
    roi_height, roi_width, _=roi.shape

    mask=masks[i,int(class_id)]
    mask=cv2.resize(mask,(roi_width,roi_height))
    _,mask=cv2.threshold(mask,0.5,255,cv2.THRESH_BINARY)

    #(image, start_point, end_point, color, thickness)
    cv2.rectangle(img,(x,y),(x2,y2),(0,255,0),2)

    #get mask coordinates
    contours,_=cv2.findContours(np.array(mask,np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    color=colors[int(class_id)]

    #fill the contours for each mask
    for cnt in contours:
        cv2.fillPoly(roi,[cnt],(int(color[0]),int(color[1]),int(color[2])))


cv2.imshow("image",img)
cv2.imshow("black_image",black_image)   
cv2.waitKey(0)