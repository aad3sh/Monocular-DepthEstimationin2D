import cv2
import os

image_folder = 'D:/DepthEstimation/outputs/'
video_name = 'D:/DepthEstimation/outputs/outputVideo.mp4'

images = ['output'+str(i)+'.png\n' for i in range(1, 270)]

frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width,height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()
