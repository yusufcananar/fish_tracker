"""
This code taken from theAIGuysCode github repository(https://github.com/theAIGuysCode/yolov3_deepsort) but edited and changed by ;
Yusuf Can Anar

 Adapted for Fish Detection, Classification, Tracking and Counting purposes.

enter below line into cmd and you can start and save Example;
python object_tracker.py --video ./data/video/test.mp4 --output ./data/video/outputX.avi --weights ./weights/yolov3-custom.tf --num_classes 2 --classes ./data/labels/obj.names
"""

import csv
import time
import numpy as np
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images


flags.DEFINE_string('classes', './data/labels/coco.names', 'path to classes file')
flags.DEFINE_string('weights', './weights/yolov3.tf',
                    'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('video', './data/video/test.mp4',
                    'path to video file or number for webcam)')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')

# custom flags
flags.DEFINE_string('record_counts', './output/counts.csv', 'path to count of detections record .csv file')
flags.DEFINE_integer('record_interval', 1, 'records count of detections every <input> seconds')

def save_to_csv(file_path, count, unique_id):
    # Check if the file exists to write headers only once
    file_exists = False
    try:
        with open(file_path, 'r'):
            file_exists = True
    except FileNotFoundError:
        file_exists = False

    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        # Write header if the file is new
        if not file_exists:
            writer.writerow(['ID', 'Date', 'Counts'])  # Write column headers
        # Write the data
        current_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        writer.writerow([unique_id, current_time, count])  # Save ID, Date, and count

def main(_argv):
    # Definition of the parameters
    font = cv2.FONT_HERSHEY_DUPLEX
    
    #initialize deep sort
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    if FLAGS.tiny:
        yolo = YoloV3Tiny(classes=FLAGS.num_classes)
    else:
        yolo = YoloV3(classes=FLAGS.num_classes)

    yolo.load_weights(FLAGS.weights)
    logging.info('weights loaded')

    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    logging.info('classes loaded')

    try:
        vid = cv2.VideoCapture(int(FLAGS.video))
    except:
        vid = cv2.VideoCapture(FLAGS.video)

    out = None

    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))
        uniqueId = 0

    startTime = 0
    fps = 0.0
    count = 0 
    while True:
        _, img = vid.read()
        
        if img is None:
            logging.warning("Empty Frame")
            time.sleep(0.1)
            count+=1
            if count < 3:
                continue
            else: 
                break

        img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        img_in = tf.expand_dims(img_in, 0)
        img_in = transform_images(img_in, FLAGS.size)
        
        # Capture frame start time
        startFrameTime = time.time()

        # Predict
        boxes, scores, classes, nums = yolo.predict(img_in)

        # map classes to object names
        classes = classes[0]
        names = []
        for i in range(len(classes)):
            names.append(class_names[int(classes[i])])
        names = np.array(names)

# *********************************************************************************************************
        # Initialize detection count
        detection_count = 0

        # Iterate through the boxes and scores and draw them if the score is greater than 0.5
        for box, score, className in zip(boxes[0], scores[0], names):  # Access the first (and only) set of boxes
            if score > 0.5:
                detection_count += 1  # Increment the detection count

                # Convert normalized coordinates to pixel coordinates
                x1 = int(box[0] * width)
                y1 = int(box[1] * height)
                x2 = int(box[2] * width)
                y2 = int(box[3] * height)
                
                # Draw the rectangle on the image
                cv2.rectangle(img, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=1)  # Blue color with thickness 2
                
                # Prepare the score text as a percentage
                score_text = f"{className} - {score * 100:.1f}%"  # Format score as percentage with one decimal place
                
                # Set the position for the text (above the rectangle)
                text_position = (x1, y1 - 10)  # Adjust the y-coordinate to position above the rectangle
                
                # Draw the text on the image
                cv2.putText(img, score_text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0),1)  # Blue color

        # Prepare the total detection count text
        total_text = f"Total: {detection_count}"

        # Draw the total detection count in the top-left corner
        cv2.putText(img, total_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)  # White color

        endFrameTime = time.time()
        # Draw fps on screen 
        fps  = ( fps + (1./(endFrameTime-startFrameTime)) ) / 2
        cv2.putText(img, "FPS: {:.2f}".format(fps), (20,int(height)-30),
                          cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (200, 0, 100), 1)
# *********************************************************************************************************

        if FLAGS.output:
            out.write(img)
            # Get the current time in seconds of the video
            endTime = vid.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # Convert milliseconds to seconds
            # Save to csv by given interval
            if(int(endTime - startTime) >= FLAGS.record_interval):
                # save current counts to csv file
                save_to_csv(FLAGS.record_counts, detection_count, uniqueId)
                uniqueId += 1
                startTime = endTime # reset start time
        
        img = cv2.resize(img,(1200,720))
        cv2.imshow('output', img)
        # press q to quit
        if cv2.waitKey(1) == ord('q'):
            break
    vid.release()
    if FLAGS.output:
        out.release()
    cv2.destroyAllWindows()

    
if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
