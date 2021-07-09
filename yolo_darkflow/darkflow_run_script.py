"""
This script is designed to detect objects using
a web camera or video. Usage example:
 $python darkflow_run_script.py
To use the video you need to add an argument
and specify the path. Usage example:
 $python darkflow_run_script.py --video PATH_TO_VIDEO
Video output will be saved in the current directory.

To change the model, you need to change all the lines marked:
TODO_MODEL_CHANGE
    pseudocode:
    1. Prepare a camera properties
    2. Load a model
    3. Starting a "main loop", which contain:
        3.1 Take a frame
        3.2 Use model to do prediction
        3.3 Transfer prediction to boxing() function to draw a boxes
    4. Do statistic
 """
# Import packages
import matplotlib.pyplot as plt
import numpy as np
import os
from darkflow.net.build import TFNet # Libriary for CNN (deep learnin)
import cv2  # OpenCV capturing video frames, working with fitters
import time
import sys
import argparse # Libriary for parsing input arguments


#Function for drawing rectangles and put a text to frame
def boxing(original_img , predictions, counter, mean_confidence):
    newImage = np.copy(original_img)
    for result in predictions:
        top_x = result['topleft']['x']
        top_y = result['topleft']['y']

        btm_x = result['bottomright']['x']
        btm_y = result['bottomright']['y']

        confidence = result['confidence']
        label = result['label'] + " " + str(round(confidence, 3))
        if confidence > 0.3:
            newImage = cv2.rectangle(newImage, (top_x, top_y), \
            (btm_x, btm_y), (255,0,0), 3)
            newImage = cv2.putText(newImage, label, (top_x, top_y-5),\
            cv2.FONT_HERSHEY_COMPLEX_SMALL , 0.8, (0, 230, 0), 1, cv2.LINE_AA)
            counter += 1
            mean_confidence.append(confidence)
    return newImage, counter, mean_confidence

# Parse input arguments from terminal
parser = argparse.ArgumentParser(description='Object detection script.')
parser.add_argument('--video', help = "PATH to video file for test" )
args = vars(parser.parse_args())

# Properties for video frame
flag_video_out = False
if not args.get("video", False):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 416)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 416)
else:
    cap = cv2.VideoCapture(args["video"])
    out = cv2.VideoWriter('out_yolo_tiny.avi',
    cv2.VideoWriter_fourcc('M','J','P','G'),30 , (640,360))
    flag_video_out = True

# Additional variables to calculate statistic
counter = 0
mean_confidence = [0]
global_time_start = time.clock()

# Custom model for recognision books
options = {"model": "cfg/tiny-yolo-voc-2c.cfg", #TODO_MODEL_CHANGE
           "load": 2700,
           "gpu": 0,
           "threshold": 0.12}

tfnet2 = TFNet(options)
tfnet2.load_from_ckpt()

# A main loop
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:
        frame = np.asarray(frame)
        results = tfnet2.return_predict(frame)
        new_frame, counter, mean_confidence = boxing(frame,
        results,counter, mean_confidence)

        # Display the resulting frame
        cv2.imshow('frame', new_frame)
        if flag_video_out == True:
            out.write(new_frame) # Write output to videofile
        if cv2.waitKey(1) & 0xFF == ord('q'): # if pressed "q" stop
            break
    else:
        break
global_time_end = time.clock()
# Print out statistic info
print("-----------Statistic Data-----------")
print("Mean FPS: ", 1260/global_time)
print("Number of bounding boxes: ", counter)
print("Mean confidence: ", np.mean(mean_confidence))
# When everything done, release the capture
cap.release()
if flag_video_out == True:
    out.release()
cv2.destroyAllWindows()
