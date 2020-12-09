#!/usr/bin/env python

import numpy as np
import cv2
import tensorflow as tf

fx = 700.895
fy = 700.895
cx = 665.54
cy = 371.155

delta_z = 50


def get_rectangles(mask, threshold_area):
    """
    Extract defined color from image and return rectangles coordinates of large enough contours on given side
    Input: 
        cv_image: Image (BGR)
        lower_range: 1x3 tuple representing lower HSV for target color
        upper_range: 1x3 tuple representing upper HSV for target color
        threshold_area: int
        side: 1 for Right, -1 for Left
    Output:
        list of 1x4 tuples (x, y, w, h) of color blobs 
    """
    contours, hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    rectangles = []
    for contour in contours:
        if cv2.contourArea(contour) > threshold_area:
            rect = cv2.boundingRect(contour)
            rectangles.append(rect)
    return rectangles


def get_contours(mask, threshold_area):
    """
    Extract defined color from image and return large contours (UNUSED)
    Input: 
        cv_image: Image (BGR)
        lower_range: 1x3 tuple representing lower HSV for target color
        upper_range: 1x3 tuple representing upper HSV for target color
        threshold_area: int
    Output:
        list of openCV contours 
    """
    contours, hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    return [x for x in contours if cv2.contourArea(x) > threshold_area], hierarchy



def color_segmentation(image, lower, upper):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
    return mask


def get_mask_pixels(mask):
    return np.transpose((mask>0).nonzero())


def calculate_distance_from_disparity(Z, v, u):
    X = (u-cx) * Z/fx
    Y = (v-cy) * Z/fy

    return (X**2 + Y**2 + Z**2)**0.5


def calculate_x_distance(disparity, x, y)
    distance_from_camera = calculate_distance_from_disparity(disparity, x, y)

    theta = np.arccos(delta_z/distance_from_camera)

    return np.sin(theta)*distance_from_camera


def get_avg_depth(depth_img, pixels, low_thres=0, high_thres=1000):
    total_depth = 0
    i = 0
    for x,y in pixels:
        disparity = depth_img[x][y]
        if disparity > low_thres and disparity < high_thres: 
            total_depth += calculate_x_distance(disparity, x, y)
            i += 1
    print("NUM PIXELS: ", i)
    return total_depth/i


def get_region_box(smask, area=100, side='bottom', image=None):
    left = mask.shape[1]
    right = 0
    top = mask.shape[0]
    bot = 0
    box = None

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) > area:
            rect = cv2.boundingRect(contour)
            if image:
                tl = (rect[0], rect[1])
                br = (rect[0]+rect[2], rect[1]+rect[3])
                cv.rectangle(image, tl, br, (255,0,0), 2)
            if side == 'left':
                if rect[0] < left:
                    left = rect[0]
                    box = rect
            elif side == 'right':
                if rect[0] > right:
                    right = rect[0]
                    box = rect
            elif side == 'top':
                if rect[1] < top:
                    top = rect[1]
                    box = rect
            else:
                if rect[1] > bot:
                    bot = rect[1]
                    box = rect
    if image:
        cv.rectangle(image, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (0,0,255), 2)
    return box


def get_tf2_detect_fn(path):
    detect_fn=tf.saved_model.load(path)
    print("LOADED MODEL!!!!!!!!!!")
    return detect_fn


def detect_objects(detect_fn, image, width=1280, height=720, min_score_thres=0.5):
    image_np = np.array(image)
    if image_np.shape[2] > 3:
        image_np = image_np[:,:,:3]
        # print(image_np)
    input_tensor=tf.convert_to_tensor(image_np)
    input_tensor=input_tensor[tf.newaxis, ...]

    detections=detect_fn(input_tensor)
    
    boxes = detections['detection_boxes'][0]
    scores = detections['detection_scores'][0]
    objects = []
    for i in range(boxes.shape[0]): 
        if scores is None or scores[i].numpy() > min_score_thres:
            # print("SCORE: ", scores[i].numpy())
            class_name = detections['detection_classes'][0][i].numpy()

            nut = boxes[i].numpy()

            for i in range(len(nut)):
                if nut[i] == np.nan:
                    print("NUT")
                    nut[i] = 0.0

            y_min, x_min, y_max, x_max = nut

            # print(nut)
            tl, br = ((int(x_min*width), int(y_min*height)), (int(x_max*width), int(y_max*height)))
            detection = {'class':class_name, 'box': (tl, br)}
            objects.append(detection)

    return objects


def get_object_depth(bag_box, rgb_image, depth_image, lower, upper):
    x1, y1 = bag_box[0]
    x2, y2 = bag_box[1] 
    crop_rgb   =   rgb_image[y1+int((y2-y1)/1.5):y2,x1:x2]
    crop_depth = depth_image[y1+int((y2-y1)/1.5):y2,x1:x2]

    mask = color_segmentation(crop_rgb, lower, upper)
    pixels = get_mask_pixels(mask)

    return get_avg_depth(crop_depth, pixels)


def check_bag_flipper_depth(detect_fn, rgb_np, depth_np, black_lower, black_upper, gray_lower, gray_upper, dist_thres=10):
    # print(rgb_np)
    print(rgb_np.shape)

    bag_box = None
    flipper_box = None

    objects = detect_objects(detect_fn, rgb_np)
    print("OBJECTS DETECTED: ", len(objects))
    for obj in objects:
        if obj['class'] == 1.0:
            if bag_box == None or bag_box[1][1] < obj['box'][1][1]:
                bag_box = obj['box']
                print("BAG DETECTED: ", bag_box)
        elif obj['class'] == 2.0:
            flipper_box = obj['box']
            print("FLIPPER DETECTED: ", flipper_box)

    # print('before: ', depth_np)

    if bag_box and flipper_box:
        # print('after: ', depth_np)
        bag_depth = get_object_depth(bag_box, rgb_np, depth_np, black_lower, black_upper)
        flipper_depth = get_object_depth(flipper_box, rgb_np, depth_np, gray_lower, gray_upper)
        print("BAG_DEPTH: ", bag_depth)
        print("FLIPPER_DEPTH: ", flipper_depth)
        if bag_depth < flipper_depth and bag_depth+dist_thres > flipper_depth:
            print("ALIGNED!")
            return True
        else:
            if bag_depth < flipper_depth:
                print("TOOO CLOSEEEEEE!!!")
            else:
                pirnt("TOOOO FARRRRRRR!!!")
    return False