import cv2
import glob


img_array = []
file_list=[]
for file_n in glob.glob('frames/*jpg'):
    file_n=file_n.replace('frames\\','')#TODO save the first and last part in variables
    file_n = file_n.replace('.jpg', '')
    file_list.append(file_n)

file_list.sort(key=int)

for filename in file_list:
    img = cv2.imread('frames\\'+filename+'.jpg')
    height, width, layers = img.shape
    size = (width, height)
    img_array.append(img)
    print('appending',filename)

out = cv2.VideoWriter('videos/project.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

for i in range(len(img_array)):
    out.write(img_array[i])
    print('Frame:',i)
out.release()

print('Done')