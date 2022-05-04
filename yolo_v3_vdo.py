import cv2
import numpy as np
import time
global cnt
cnt=0

def load_yolo():

    net=cv2.dnn.readNetFromDarknet("yolov3_testing.cfg","yolov3.weights")

    with open("coco.names", "r") as f:
      classes = [line.strip() for line in f.readlines()]

    output_layers = [layer_name for layer_name in net.getUnconnectedOutLayersNames()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    return net, classes, colors, output_layers


def start_video(video_path):
   model, classes, colors, output_layers = load_yolo()
   cap = cv2.VideoCapture(video_path)
   cnte=0
   # out = cv2.VideoWriter('videos/proc.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (416,416))
   while True:
      ret, frame = cap.read()
      print(ret,cnte)
      # cv2.imshow('Win',frame)
      # cv2.waitKey(1500)

      height, width, channels = frame.shape
      blob, outputs = detect_objects(frame, model, output_layers)
      boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
      draw_labels(boxes, confs, colors, class_ids, classes, frame)
      cnte+=1
      # out.write(img) #--- out----FUNC
      key = cv2.waitKey(1)
      if key == 27:
         break
   cap.release()




def get_box_dimensions(outputs, height, width):
   boxes = []
   confs = []
   class_ids = []
   prev_frame_time=0
   new_prev_frame_time=0


   for output in outputs:
      for detect in output:
         scores = detect[5:]
         class_id = np.argmax(scores)
         conf = scores[class_id]
         if conf > 0.4:   #Try .6
            center_x = int(detect[0] * width)
            center_y = int(detect[1] * height)
            w = int(detect[2] * width)
            h = int(detect[3] * height)
            x = int(center_x - w/2)
            y = int(center_y - h / 2)
            boxes.append([x, y, w, h])
            confs.append(float(conf))
            class_ids.append(class_id)
   # codec = cv2.VideoWriter_fourcc(*"MJPG")
   # new_frame_time = time.time()
   # fps = 1 / (new_frame_time - prev_frame_time)
   # prev_frame_time = new_frame_time
   # fps_n = int(fps)

   return boxes, confs, class_ids

def detect_objects(img, net, outputLayers):
    blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(320, 320), mean=(0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(outputLayers)
    return blob, outputs



def draw_labels(boxes, confs, colors, class_ids, classes, img):
   global cnt
   indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4) #(0.2,0.2) // (0.7,0.6)
   font = cv2.FONT_HERSHEY_PLAIN
   # image_folder = 'data-set-race-01'
   video_file = 'videos/proc.mp4'
   image_size = (416, 416)
   fps = 24
   # out = cv2.VideoWriter(video_file, cv2.VideoWriter_fourcc('M','P','E','G'), fps, image_size)

   for i in range(len(boxes)):
      if i in indexes:
         x, y, w, h = boxes[i]
         label = str(classes[class_ids[i]])
         color = colors[i]
         cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
         cv2.putText(img, label, (x, y - 5), font, 1, color, 1)
         # out.write(img)

   cv2.imshow("Image", img)
   cv2.imwrite('frames/'+str(cnt)+'.jpg',img)
   print('\t',cnt)
   cnt += 1


#----------------------------------MAIN -------------------
# if _name_ == '_main_':
video_path = 'videos/test_Drone.mp4'
print('Opening ' + video_path + " .... ")
start_video(video_path)
print('Done')