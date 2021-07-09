"""
This script is designed to detect objects using
a web camera or video. Usage example:
 $python tensorflow_run_sctipt.py
To use the video you need to add an argument
and specify the path. Usage example:
 $python tensorflow_run_sctipt.py --video PATH_TO_VIDEO
Video output will be saved in the current directory.

To change the model, you need to change all the lines marked:
TODO_MODEL_CHANGE


    pseudocode:
    1. Path to all files
    2. Load label maps
    3. Load model into memory
    4. Define input and output tensors for all prediction data
    5. Parse input arguments
    6. Prepare a camera properties
    7. Starting a "main loop", which contain:
        7.1 Take a frame
        7.2 Use model to do prediction
        7.3 Visualize with
        visualize_boxes_and_labels_on_image_array() function
    8. Do statistic
    9. Clean up

 """
# Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import argparse
import time
# Import utilites Tensorflow API
from utils import label_map_util
from utils import visualization_utils as vis_util

# Name of the directory containing the object detection path
MODEL_NAME = 'books_graph_ssd' #TODO_MODEL_CHANGE
# Path to frozen detection graph .pb file
PATH_TO_CKPT = os.path.join(os.getcwd(),MODEL_NAME,
'frozen_inference_graph.pb') #TODO_MODEL_CHANGE
# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'training_ssd',
'labelmap.pbtxt')
# Number of classes the object detector can identify
NUM_CLASSES = 2
# Load the label map.
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map,
max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
    sess = tf.Session(graph=detection_graph)

# Define input and output tensors
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

parser = argparse.ArgumentParser(description='Object detection script.')
parser.add_argument('--video', help = "PATH to video file for test" )
args = vars(parser.parse_args())
# Properties for video frame
flag_video_out = False
if not args.get("video", False):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 300)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 300)
else:
    cap = cv2.VideoCapture(args["video"])
    out = cv2.VideoWriter('out_ssd.avi',\
    cv2.VideoWriter_fourcc('M','J','P','G'), 30, (640,360))
    flag_video_out = True

# Additional variables to calculate statistic
counter = 0
mean_accurasy = [0]
global_time_start = time.clock()
# Main loop
while(True):
    # Capture the frame
    ret, frame = cap.read()
    if ret == False:
        break
    frame_expanded = np.expand_dims(frame, axis=0)
    # Run the model with the image as input
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: frame_expanded})
    # Draw the results of the detection
    vis_util.visualize_boxes_and_labels_on_image_array(
        frame,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8,
        min_score_thresh=0.60)

    for i in np.squeeze(scores):
         if i > 0.2:
            mean_accurasy.append(i)

    # Display the results
    cv2.imshow('Object detector', frame)
    if flag_video_out == True:
        out.write(frame)# Write output to videofile
    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Print out statistic info
global_time_end = time.clock()
print("-----------Statistic Data-----------")
print("FPS: ", 1260.0/(global_time_end-global_time_start))
print("Number of bounding boxes: ", len(mean_accurasy))
print("mean accurasy: ", np.mean(mean_accurasy))
# Clean up
cap.release()
if flag_video_out == True:
    out.release()
cv2.destroyAllWindows()
