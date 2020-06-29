"""
https://github.com/nwojke/deep_sort
@inproceedings{Wojke2017simple,
  title={Simple Online and Realtime Tracking with a Deep Association Metric},
  author={Wojke, Nicolai and Bewley, Alex and Paulus, Dietrich},
  booktitle={2017 IEEE International Conference on Image Processing (ICIP)},
  year={2017},
  pages={3645--3649},
  organization={IEEE},
  doi={10.1109/ICIP.2017.8296962}
}

@inproceedings{Wojke2018deep,
  title={Deep Cosine Metric Learning for Person Re-identification},
  author={Wojke, Nicolai and Bewley, Alex},
  booktitle={2018 IEEE Winter Conference on Applications of Computer Vision (WACV)},
  year={2018},
  pages={748--756},
  organization={IEEE},
  doi={10.1109/WACV.2018.00087}
}

This code taken from theAIGuysCode github repository(https://github.com/theAIGuysCode/yolov3_deepsort) but edited and changed by ;
Yusuf Can Anar
Kübra Traş

 Adapted for Fish Detection, Classification, Tracking and Counting purposes.

enter below line into cmd and you can start and save Example;
python object_tracker.py --video ./data/video/test.mp4 --output ./data/video/outputX.avi --weights ./weights/yolov3-custom.tf --num_classes 2 --classes ./data/labels/obj.names
"""

import time, random
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
from yolov3_tf2.utils import draw_outputs, convert_boxes

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from PIL import Image


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

dict_tracks = {"Koi":{}, "Tilapia":{}}
def get_patterns(center,track_id,class_name):
    #This function stores all tracked fish and their moving patterns
    if class_name == 'Koi':

        if str(track_id) in dict_tracks["Koi"]:
            dict_tracks["Koi"][str(track_id)].append(center)
        elif str(track_id) not in dict_tracks["Koi"]:
            dict_tracks["Koi"][str(track_id)] = []
            dict_tracks["Koi"][str(track_id)].append(center)
        if len(dict_tracks["Koi"][str(track_id)]) > 60:
            del dict_tracks["Koi"][str(track_id)][:10]
            
        return dict_tracks["Koi"][str(track_id)]
    
    elif class_name == 'Tilapia':
        if str(track_id) in dict_tracks["Tilapia"]:
            dict_tracks["Tilapia"][str(track_id)].append(center)
        elif str(track_id) not in dict_tracks["Tilapia"]:
            dict_tracks["Tilapia"][str(track_id)] = []
            dict_tracks["Tilapia"][str(track_id)].append(center)
        if len(dict_tracks["Tilapia"][str(track_id)]) > 60:
            del dict_tracks["Tilapia"][str(track_id)][:10]
        
        return dict_tracks["Tilapia"][str(track_id)]
        
def main(_argv):
    # Definition of the parameters
    right2left_koi = 0
    right2left_til = 0
    left2right_koi = 0
    left2right_til = 0
    font = cv2.FONT_HERSHEY_DUPLEX
    max_cosine_distance = 0.5
    nn_budget = None
    nms_max_overlap = 1.0
    
    #initialize deep sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

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
        list_file = open('detection.txt', 'w')
        frame_index = -1 
    
    #midline position variables
    midline_pos_x = int(width/2) - 3
    midline_pos_y = int(height)
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
        
        t1 = time.time()
        boxes, scores, classes, nums = yolo.predict(img_in)
        classes = classes[0]
        names = []
        for i in range(len(classes)):
            names.append(class_names[int(classes[i])])
        names = np.array(names)
        converted_boxes = convert_boxes(img, boxes[0])
        features = encoder(img, converted_boxes)    
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(converted_boxes, scores[0], names, features)]
        
        #initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # run non-maxima suppresion
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]        

        # Call the tracker
        tracker.predict()
        tracker.update(detections)
        
        #draw midline
        cv2.line(img,(midline_pos_x,0),(midline_pos_x,midline_pos_y),(0,0,0),3)
        
        screen1_koi = 0
        screen1_til = 0
        screen2_koi = 0
        screen2_til = 0
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            bbox = track.to_tlbr()
            class_name = track.get_class()
            c_curr = (int(bbox[0]+abs(bbox[0]-bbox[2])/2), int(bbox[1]+abs(bbox[1]-bbox[3])/2))
            center_x = c_curr[0]
            color = colors[int(track.track_id) % len(colors)]
            color = [i * 255 for i in color]
            cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 1)
            cv2.rectangle(img, (int(bbox[0]), int(bbox[1]-17)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*14, int(bbox[1])), color, -1)
            cv2.putText(img, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-1)),font, 0.6, (0,0,0),1)
            
            #store patterns of individual fish
            pattern = get_patterns(c_curr,track.track_id,class_name)
            pre_p = c_curr
            #Draw the patterns on the screen
            for p in pattern[-50::5]:
                cv2.circle(img,p,3,color,-1)
                if pre_p != c_curr:
                    cv2.line(img,pre_p,p,color,1)
                pre_p = p
            
            if len(pattern) >= 2:
                moving2right = center_x > pattern[-2][0]
                on_screen_left = pattern[-2][0] < midline_pos_x
                moving2left = center_x < pattern[-2][0]
                on_screen_right = pattern[-2][0] > midline_pos_x
                
                if (class_name == 'Koi') and on_screen_left:
                    screen1_koi += 1
                    if moving2right and center_x > midline_pos_x:
                        left2right_koi += 1
                if (class_name == 'Tilapia') and on_screen_left:
                    screen1_til += 1
                    if moving2right and center_x > midline_pos_x:
                        left2right_til += 1
                if (class_name == 'Koi') and on_screen_right:
                    screen2_koi += 1
                    if moving2left and center_x < midline_pos_x:
                        right2left_koi += 1
                if (class_name == 'Tilapia') and on_screen_right:
                    screen2_til += 1 
                    if moving2left and center_x < midline_pos_x:
                        right2left_til += 1
                
        
        ### UNCOMMENT BELOW IF YOU WANT CONSTANTLY CHANGING YOLO DETECTIONS TO BE SHOWN ON SCREEN
        # for det in detections:
        #     bbox = det.to_tlbr() 
        #     cv2.rectangle(img,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,0,0), 2)
        
        #Print the instantenous numbers detected on the screens
        cv2.putText(img,"Koi : "+str(screen1_koi),(20,30),font,0.7,(30,50,205),2)
        cv2.putText(img,"Tla : "+str(screen1_til),(20,70),font,0.7,(255,0,0),2)
    
        cv2.putText(img,"Koi : "+str(screen2_koi),(int(width)-120,30),font,0.7,(30,50,205),2)
        cv2.putText(img,"Tla : "+str(screen2_til),(int(width)-120,70),font,0.7,(255,0,0),2)
        
        #Print left2right and right2left counts and total of them
        cv2.putText(img,str(right2left_koi)+" <-- Koi",(midline_pos_x-75,int(height)-30),font,0.7,(0,0,0),2)
        cv2.putText(img,str(right2left_til)+" <-- Tla",(midline_pos_x-75,int(height)-70),font,0.7,(0,0,0),2)
        
        cv2.putText(img,"Koi  --> " + str(left2right_koi),(midline_pos_x-55,30),font,0.7,(0,0,0),2)
        cv2.putText(img,"Tla  --> " + str(left2right_til),(midline_pos_x-55,70),font,0.7,(0,0,0),2)
        
        cv2.putText(img,"Total L2R : " + str(left2right_koi+left2right_til),(int(width)-200,int(height)-30),font,0.7,(0,0,0),2)
        cv2.putText(img,"Total R2L : " + str(right2left_koi+right2left_til),(int(width)-200,int(height)-70),font,0.7,(0,0,0),2)
        
        # print fps on screen 
        fps  = ( fps + (1./(time.time()-t1)) ) / 2
        cv2.putText(img, "FPS: {:.2f}".format(fps), (20,int(height)-30),
                          cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (200, 0, 100), 2)
        
        if FLAGS.output:
            out.write(img)
            frame_index = frame_index + 1
            list_file.write(str(frame_index)+' ')
            if len(converted_boxes) != 0:
                for i in range(0,len(converted_boxes)):
                    list_file.write(str(converted_boxes[i][0]) + ' '+str(converted_boxes[i][1]) + ' '+str(converted_boxes[i][2]) + ' '+str(converted_boxes[i][3]) + ' ')
            list_file.write('\n')
        
        img = cv2.resize(img,(1200,720))
        cv2.imshow('output', img)
        # press q to quit
        if cv2.waitKey(1) == ord('q'):
            break
    vid.release()
    if FLAGS.output:
        out.release()
        list_file.close()
    cv2.destroyAllWindows()

    
if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
