
"""TODO
    1. Convert Picture to video
    2. Rectify bounding box
    3. Counting the number of instances
    4. Analysis??
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np

net=cv2.dnn.readNetFromDarknet('yolov3_testing.cfg','yolov3.weights')

with open('coco.names','r') as f:
    classes=[line.strip() for line in  f.readlines()]
print(classes)


img_path='C:\Vedant_\Projects\PyCharm_Proj\Drone\images\img_apple.jpg'
my_img= cv2.imread(img_path) #raw image

# print(wt,ht)
img=cv2.resize(my_img,(416,416))#TODO Use Scale Factor and bring back to original #resized image
wt,ht,z = img.shape #width and height of resized (416,416)
og_img= img.copy()
# wt = wt * 0.4
# ht = ht * 0.4

blob = cv2.dnn.blobFromImage(img,1/255,(416,416),(0,0,0),True,crop=False)

# for i in blob:
#     for n, blob_img in enumerate(i): #TODO explain
#         cv2.imshow(str(n),blob_img)
#
# # cv2.waitKey(0)

net.setInput(blob)
last_lay=net.getUnconnectedOutLayersNames()
out_lay=net.forward(last_lay)
# print(out_lay[0][0]) TODO explain

box=[]
cnfi=[] #confidence
classId=[]
cnt=0 #counter


for out in out_lay:
    for det in out:
        score = det[5:] # For taking in class_IDs
        ci= np.argmax(score)#temp Class ID
        cf=score[ci]
        # print('score:',score,'\n',cnt)

        if cf > 0.5:
            cent_x=int(det[0]* wt)
            cent_y=int(det[1]* ht)

            w = int(det[2] * wt)
            h = int(det[3] * ht)
            x = int(cent_x - w / 2)
            y = int(cent_y - h / 2)

            #PRINTING
            # cv2.circle(img,(cent_x,cent_y), 10, (15,255,0),2) #TODO Error in printing the circle
            # cv2.rectangle(img,(x,y),(x+w,y+h), (255,0,0),2)
            # print("Found:",classes[ci],cnt)
            # print("at:",cent_x, cent_y)

            cnfi.append(float(cf))
            box.append([x,y,w,h])
            classId.append(ci)
            # print(box[cnt])
            # print(cnfi[cnt]*100,'\n')





#---------------------------------Loop Ends-------------------------------------

index = cv2.dnn.NMSBoxes(box,cnfi,0.5,0.5)
print(index)



#------------------------------Printing Part------------------------------------
nod=len(box) #number of objects detetced

for i in index:
    x,y,w,h =box[i]
    lbl=str(classes[classId[i]])
    # cv2.circle(img, (cent_x, cent_y), 10, (15, 255, 0), 2)  # TODO Error in printing the circle
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    lbl=lbl+str(int(cnfi[i] * 100))
    cv2.putText(img,lbl,(x,y+15),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
    print("Found:", classes[classId[i]], cnt)
    print("at:",x,y)
    print("Accuracy:",cnfi[i]*100)
    cnt+=1


# cv2.circle(img,(686,400), 10, (15,255,0),2) #TODO Error in printing the circle

cv2.imshow('image',img)
cv2.imshow('Original',og_img)
cv2.waitKey(0) #TODO try to print using plt
cv2.destroyAllWindows()
print("Reached End")
