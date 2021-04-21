import numpy as np
import cv2
from PIL import Image
import config as cfg

cap = cv2.VideoCapture("D:/DepthEstimation/inputs/video.mp4")
i = 0
while cap.isOpened():
    ret, frame = cap.read()
    if ret == True:
        im = Image.fromarray(frame)
        i += 1
        im.save('D:/DepthEstimation/inputs/video/input'+str(i)+'.png', format='png')
    else:
        print("Done processing video")
        break
i = 270
file = open(cfg.FLAGS.filenames_path, mode='a+')
for j in range(1, i+1):
    if j > 1:
        file.write("\n")
    file.write(cfg.FLAGS.home_path+'/inputs/video/input'+str(j)+'.png,'+cfg.FLAGS.home_path+'/outputs/output'+str(j)+'.png', )
file.close()
