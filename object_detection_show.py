import numpy as np
from PIL import Image
import warnings

import tensorflow as tf
import time
import cv2


PATH_TO_SAVED_MODEL="Object_Detection/exported-models/my_model1/saved_model"
print('Loading model...')# Load saved model and build the detection function
detect_fn=tf.saved_model.load(PATH_TO_SAVED_MODEL)
print('Done!')

warnings.filterwarnings('ignore')

# category_index=label_map_util.create_category_index_from_labelmap("/home/mattrix/oyster_flipper/Object_Detection/annotations/label_map.pbtxt",use_display_name=True)

img = ['Object_Detection/images/test/right000160.png']

for image_path in img:
  print('Running inference for {}... '.format(image_path))
  image = Image.open(image_path)
  image_np=np.array(image)
  if image_np.shape[2] > 3:
    print("AHHHHH")
    print(image_np.shape)
    image_np = image_np[:,:,:3]

  input_tensor=tf.convert_to_tensor(image_np)
  input_tensor=input_tensor[tf.newaxis, ...]
  detections=detect_fn(input_tensor)
  print(type(detections))
  num_detections=int(detections.pop('num_detections'))
  print(num_detections)

  # This is the way I'm getting my coordinates
  boxes = detections['detection_boxes'][0]
  print(boxes)
  # get all boxes from an array
  max_boxes_to_draw = boxes.shape[0]
  # get scores to get a threshold
  scores = detections['detection_scores'][0]
  print(scores)
  # this is set as a default but feel free to adjust it to your needs
  min_score_thresh=.5
  # iterate over all objects found
  image_np = np.array(image_np)
  for i in range(min(max_boxes_to_draw, boxes.shape[0])):
      # 
      if scores is None or scores[i] > min_score_thresh:
          # boxes[i] is the box which will be drawn
          class_name = detections['detection_classes'][0][i].numpy()
          if class_name == 1.0:
            print("BAG DETECTED")
          elif class_name == 2.0:
            print("FLIPPER DETECTED")

          # print("NUT: ", class_name)
          y_min, x_min, y_max, x_max = boxes[i].numpy()
          print("BOX: ", [x_min, x_max, y_min, y_max])

          height = 720
          width = 1280
          tl, br = ((int(x_min*width), int(y_min*height)), (int(x_max*width), int(y_max*height)))
          print(type(image_np))
          cv2.rectangle(image_np, tl, br, (255,0,0), 3)

  cv2.rectangle(image_np, (0, 100), (100,200), (0,255,0), 3)
  cv2.imshow('nut',image_np)
  cv2.waitKey(0)