import util
import cv2

path = '/home/oyster/oyster_sweep2/'

behind_image = path + 'left000594.png'
behind_depth = path + 'depth000594.png'

align_image = path + 'left000650.png'
align_depth = path + 'depth000650.png'

front_image = path + 'left000700.png'
front_depth = path + 'depth000700.png'

# img = [(behind_image, behind_depth), (align_image, align_depth), (front_image, front_depth)]
img = [(behind_image, behind_depth)]

modelpath = "/home/oyster/TensorFlow/workspace/oyster_flipper/exported-models/my_model/saved_model"

print(cv2.IMREAD_ANYDEPTH)

detect_fn = util.get_tf2_detect_fn(modelpath)

GRAY_LOWER_RANGE  = (0,5,0)
GRAY_UPPER_RANGE  = (100,10,100)
BLACK_LOWER_RANGE = (0,0,0)
BLACK_UPPER_RANGE = (360,255,40)

for img_path, depth_path in img:
    print(img_path)
    image = cv2.imread(img_path)
    print(image[0:10][0:10])
    depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
    # print(depth)

    aligned = util.check_bag_flipper_depth(detect_fn, image, depth, BLACK_LOWER_RANGE, BLACK_UPPER_RANGE, GRAY_LOWER_RANGE, GRAY_UPPER_RANGE)
    print(aligned)