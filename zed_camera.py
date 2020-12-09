import pyzed.sl as sl
import math
import numpy as np
import sys
import rospy
from std_msgs.msg import UInt16

import util


path = "/home/oyster/TensorFlow/workspace/oyster_flipper/exported-models/my_model/saved_model"
pub_topic = "/boat_controller/flip/arduino/start"
detect_fn = util.get_tf2_detect_fn(path)
flip_pub = rospy.Publisher(pub_topic, UInt16, queue_size=5)
status_pub = rospy.Publisher('status', UInt16, queue_size=5)
counter = 0

GRAY_LOWER_RANGE  = (0,5,0)
GRAY_UPPER_RANGE  = (100,10,100)
BLACK_LOWER_RANGE = (0,0,0)
BLACK_UPPER_RANGE = (360,255,40)


def main():
    # Create a Camera object
    zed = sl.Camera()

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE  # Use PERFORMANCE depth mode
    init_params.coordinate_units = sl.UNIT.CENTIMETER  # Use meter units (for depth measurements)
    init_params.camera_resolution = sl.RESOLUTION.HD720

    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        exit(1)

    # Create and set RuntimeParameters after opening the camera
    runtime_parameters = sl.RuntimeParameters()
    runtime_parameters.sensing_mode = sl.SENSING_MODE.STANDARD  # Use STANDARD sensing mode
    # Setting the depth confidence parameters
    runtime_parameters.confidence_threshold = 100
    runtime_parameters.textureness_confidence_threshold = 100

    # Capture 150 images and depth, then stop
    i = 0
    image = sl.Mat()
    depth = sl.Mat()

    mirror_ref = sl.Transform()
    mirror_ref.set_translation(sl.Translation(2.75,4.0,0))
    tr_np = mirror_ref.m

    finding = True

    while finding and not rospy.is_shutdown():
        # A new image is available if grab() returns SUCCESS
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            # Retrieve left image
            zed.retrieve_image(image, sl.VIEW.LEFT)
            # Retrieve depth map. Depth is aligned on the left image
            zed.retrieve_measure(depth, sl.MEASURE.DEPTH)

            rgb_np = np.array(image.get_data())
            depth_np = np.array(depth.get_data())

            aligned = util.check_bag_flipper_depth(detect_fn, rgb_np, depth_np, BLACK_LOWER_RANGE, BLACK_UPPER_RANGE, GRAY_LOWER_RANGE, GRAY_UPPER_RANGE)
            if aligned:
                counter += 1
                if counter >= 10:
                    print("FLIPPING")
                    pub.publish(0)
                    finding = False
                    counter = 0
            else:
                counter = 0
        status_pub.publish(0)

    # Close the camera

    zed.close()


if __name__ == "__main__":
    rospy.init_node('main', anonymous=True)
    try:
        main()
    except rospy.ROSInterruptException:
        pass

