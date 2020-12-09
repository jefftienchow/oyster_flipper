import sys
import pyzed.sl as sl
import cv2
import numpy as np
import cv2
from pathlib import Path
import enum

import util


filepath = '/home/oyster/sweep2.svo'
path = "/home/oyster/TensorFlow/workspace/oyster_flipper/exported-models/my_model/saved_model"

test_rgb_path = '/home/oyster/test_rgb594.png'
test_depth_path = '/home/oyster/test_depth594.png'


detect_fn = util.get_tf2_detect_fn(path)

GRAY_LOWER_RANGE  = (0,5,0)
GRAY_UPPER_RANGE  = (100,20,100)
BLACK_LOWER_RANGE = (0,0,0)
BLACK_UPPER_RANGE = (360,255,40)

path = '/home/oyster/oyster_sweep2/'

def main():
    print("Reading SVO file: {0}".format(filepath))

    init_params = sl.InitParameters()
    init_params.set_from_svo_file(str(filepath))
    init_params.svo_real_time_mode = False  # Don't convert in realtime
    init_params.coordinate_units = sl.UNIT.CENTIMETER 
    # Create ZED objects
    zed = sl.Camera()

    # Open the SVO file specified as a parameter
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        sys.stdout.write(repr(err))
        zed.close()
        exit()
    
    # Get image size
    image_size = zed.get_camera_information().camera_resolution
    width = image_size.width
    height = image_size.height
    width_sbs = width * 2
    
    # Prepare side by side image container equivalent to CV_8UC4
    svo_image_sbs_rgba = np.zeros((height, width_sbs, 4), dtype=np.uint8)

    # Prepare single image containers
    image = sl.Mat()
    depth = sl.Mat()
    rt_param = sl.RuntimeParameters()

    # Start SVO conversion to AVI/SEQUENCE
    sys.stdout.write("Converting SVO... Use Ctrl-C to interrupt conversion.\n")

    nb_frames = zed.get_svo_number_of_frames()

    while True:
        if zed.grab(rt_param) == sl.ERROR_CODE.SUCCESS:
            svo_position = zed.get_svo_position()

            # Retrieve SVO images
            zed.retrieve_image(image, sl.VIEW.LEFT)

            zed.retrieve_measure(depth, sl.MEASURE.DEPTH)

            if svo_position < nb_frames:
                rgb_np = image.get_data()
                depth_np = depth.get_data()

                aligned = util.check_bag_flipper_depth(detect_fn, rgb_np, depth_np, BLACK_LOWER_RANGE, BLACK_UPPER_RANGE, GRAY_LOWER_RANGE, GRAY_UPPER_RANGE)
                print(aligned)

                print('PROGRESS: ', svo_position)
            # Check if we have reached the end of the video
            if svo_position >= (nb_frames - 10):  # End of SVO
                sys.stdout.write("\nSVO end has been reached. Exiting now.\n")
                break

    zed.close()
    return 0


if __name__ == "__main__":
    main()
