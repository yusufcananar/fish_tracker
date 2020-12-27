# Fish Detection, Tracking and Counting using YOLOv3 and DeepSORT

  This repository implements YOLOv3 and DeepSORT for tracking and counting of 2 different fish species in an aquarium.
YOLO(You only look once) uses CNN to detect objects in real time. A single neural network is applied to the full image by the algorithm and the image is divided into regions, predicts bounding boxes and the probabilities for each region. Then the predicted probabilities weight the bounding boxes. We can use these detections to feed DeepSORT(Simple Online and Realtime Tracking with a Deep Association Metric) in order to track the fish. Main purpose of this repository after tracking the fish, to store the fish movements in other word behaviors and to count the fish going one place to another. <Enter>
  
![Fish Tracking and Counting](data/gifs/output14-gif.gif)

##### You can check out the youtube video of the result on the link below:
<a href="https://www.youtube.com/watch?v=Fu2W3UVwYIE
" target="_blank"><img src="http://img.youtube.com/vi/Fu2W3UVwYIE/0.jpg" 
alt="Fish Tracking and Counting" width="240" height="180" border="10" /></a>

# How to apply?
I recommend building the tracker in an anaconda environment. So, download anaconda. Then open anaconda prompt as administrator.
Go to your path(location of the unzipped tracker file). Create an environment named as tracker-gpu (if you do not have a gpu you can name it as tracker-cpu). And download the dependencies in the conda-gpu.yml file(or conda-cpu.yml). Activate the tracker-gpu environment.
```
conda env create -f conda-gpu.yml
conda activate tracker-gpu
```
The code below will convert the yolov3 weights into TensorFlow .tf model files. Make sure that your weights in the weights folder.
```
# yolov3-custom (add --tiny flag if your custom weights were trained for tiny model)
python load_weights.py --weights ./weights/<YOUR CUSTOM WEIGHTS FILE> --output ./weights/yolov3-custom.tf --num_classes <# CLASSES>
```
Enter the below lines to your anaconda prompt to be able run the fish tracker.
```
#yolov3-custom (add --tiny flag if your custom weights were trained for tiny model)
python object_tracker.py --video ./data/video/your_test_video.mp4 --output ./data/video/results.avi --weights ./weights/yolov3-custom.tf --num_classes <# CLASSES> --classes ./data/labels/<YOUR CUSTOM .names FILE>
```
#### I have not build a repository for training part of this project but you can check out theAIGuysCode youtube channel https://www.youtube.com/watch?v=10joRJt39Ns

# References

[DeepSORT Repository](https://github.com/nwojke/deep_sort) <Enter>
  
[YoloV3 Implemented in Tensorflow 2.0](https://github.com/zzh8829/yolov3-tf2) <Enter>
  
[theAIGuysCode- Great Tutorial for Object Tracking Implementation](https://github.com/theAIGuysCode/yolov3_deepsort) <Enter>
  
 
 
