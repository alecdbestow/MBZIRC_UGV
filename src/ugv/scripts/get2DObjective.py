#!/usr/bin/env python3

import debugpy

debugpy.listen(5678)

debugpy.wait_for_client()


import numpy as np
import tensorflow as tf
import rospy
import cv2
import time

from ugv.srv import get2DObjective, get2DObjectiveResponse, saveImages
from ugv.msg import target
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from IPython.display import display

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1

# Patch the location of gfile
tf.gfile = tf.io.gfile
DATE = 1
DROP_BOX = 2

class ObjectDetector:
    def __init__(self):
        self.PATH_TO_LABELS = '/home/alec/Documents/Tensorflow/workspace/training_demo/annotations/label_map.pbtxt'
        self.category_index = label_map_util.create_category_index_from_labelmap(self.PATH_TO_LABELS, use_display_name=True)

        self.model_name = 'my_new_model2'
        self.model_dir = "/home/alec/Documents/Tensorflow/workspace/training_demo/exported-models/my_model/saved_model"

        self.model = tf.saved_model.load(str(self.model_dir))

    def run_inference_for_single_image(self, image):
        # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
        input_tensor = tf.convert_to_tensor(image)
        # The model expects a batch of images, so add an axis with `tf.newaxis`.
        input_tensor = input_tensor[tf.newaxis,...]

        # Run inference
        model_fn = self.model.signatures['serving_default']
        output_dict = model_fn(input_tensor)

        # All outputs are batches tensors.
        # Convert to numpy arrays, and take index [0] to remove the batch dimension.
        # We're only interested in the first num_detections.
        num_detections = int(output_dict.pop('num_detections'))
        output_dict = {key:value[0, :num_detections].numpy() 
                        for key,value in output_dict.items()}
        output_dict['num_detections'] = num_detections

        # detection_classes should be ints.
        output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

        # Handle models with masks:
        if 'detection_masks' in output_dict:
        # Reframe the the bbox mask to the image size.
            detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                        output_dict['detection_masks'], output_dict['detection_boxes'],
                        image.shape[0], image.shape[1])      
            detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                                tf.uint8)
            output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
        
        image_np = image.copy()
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            self.category_index,
            instance_masks=output_dict.get('detection_masks_reframed', None),
            use_normalized_coordinates=True,
            line_thickness=8)

        
        
        cv2.imwrite('/home/alec/Pictures/cv_images/' + str(time.time_ns()) + '.jpg', image_np)
    
        
        
        
        
        
        return output_dict

    def show_inference(self, image):
        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        # Actual detection.
        image_np = image.copy()
        #self.image.setflags(write=1)
        #image_np.setflags(write=1)
        output_dict = self.run_inference_for_single_image(image_np)


        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            self.category_index,
            instance_masks=output_dict.get('detection_masks_reframed', None),
            use_normalized_coordinates=True,
            line_thickness=8)

        
        
        cv2.imwrite('/home/alec/Pictures/cv_images/' + str(time.time_ns()) + '.jpg', image_np)
    
 

class Point:
    def __init__(self, X, Y):
        self.x = X
        self.y = Y

    def __add__(self, p1):
        return Point(self.x + p1.x, self.y + p1.y)
    
    def __truediv__(self, div):
        return Point(self.x / div, self.y / div)

class Box:
    def __init__(self, p1, p2) -> None:
        self.lower = p1
        self.upper = p2
    
    def addNumberOffset(self, x, y):
        self.lower.x += x
        self.upper.x += x
        self.lower.y += y
        self.upper.y += y

    def addOffset(self, tile):
        self.lower = self.lower + tile.lower
        self.upper = self.upper + tile.lower
    
    def getMiddle(self):
        return Point(
            (self.lower.x + self.upper.x)/2, 
            (self.lower.y + self.upper.y)/2
        )
        
        

class Tile:
    def __init__(self, pos, image):

        height = image.shape[0]
        width = image.shape[1]
        
        # Tile positions are:
        # 0 1
        # 2 3
        if pos == 0:
            self.lower = Point(0,0)
            self.upper = Point(width // 2, height // 2)
        elif pos == 1:
            self.lower = Point(width // 2,0)
            self.upper = Point(width, height // 2)
        elif pos == 2:
            self.lower = Point(0, height // 2)
            self.upper = Point(width // 2, height)
        elif pos == 3:
            self.lower = Point(width // 2, height // 2)
            self.upper = Point(width, height)
        
        self.tile = image[self.lower.y:self.upper.y, self.lower.x:self.upper.x]

def getTiles(image):
    return [Tile(0, image), Tile(1, image), Tile(2, image), Tile(3, image)]

class Detector:
    def __init__(self):
        self.od = ObjectDetector()

        
    def getDateList(self):
        self.od.show_inference(self.image)

    def getColourImageCallback(self, image_message):
        bridge = CvBridge()
        cv_image = bridge.imgmsg_to_cv2(image_message, desired_encoding='bgr8')
        
        self.image = np.asarray_chkfinite(cv_image)

    def getDepthImageCallback(self, depth_message):
        self.depth_image = Image()
        self.depth_image = depth_message
        

    def getDepthAtPoint(self, depth_message, point):
        row_step = depth_message.step / depth_message.width
        pos = int(row_step * point.x + point.y * depth_message.step)
        return int.from_bytes(
            [depth_message.data[pos], depth_message.data[pos + 1]], 
            byteorder='little', 
            signed=False
        )

    def getBoxDisparity(self, box):
        max = 0
        min = 0
        for y in range(int(box.lower.y), int(box.upper.y)):
            for x in range(int(box.lower.x), int(box.upper.x)):
                dist = self.getDepthAtPoint(self.depth_image, Point(x, y))
                if dist > 0:
                    if dist < min:
                        min = dist
                    elif dist > max:
                        max = dist
        
        return max - min

    def getboxMinDistance(self, box):
        min = 9999999
        for y in range(int(box.lower.y), int(box.upper.y)):
            for x in range(int(box.lower.x), int(box.upper.x)):
                dist = self.getDepthAtPoint(self.depth_image, Point(x, y))
                if  0 < dist < min:
                    min = dist     
        
        return min


        
    def extractBoxes(self, output_dict, objective):

        tempTiles = getTiles(
            cv2.resize(
                self.image, 
                (790, 445), #enjoy the magic numbers
                interpolation = cv2.INTER_AREA)
        )
        height = tempTiles[0].tile.shape[0]
        width = tempTiles[0].tile.shape[1]
        

        MIN_CONFIDENCE = 0
        if (objective == DATE):
            MIN_CONFIDENCE = 0.8
        else:
            MIN_CONFIDENCE = 0.40
        boxes = []
        i = 0
        # The detection boxes are ordered by confidence
        while output_dict['detection_scores'][i] > MIN_CONFIDENCE:
            if output_dict['detection_classes'][i] == objective:


                temp = output_dict['detection_boxes'][i]
                lower = Point(temp[1] * width, temp[0] * height)
                upper = Point(temp[3] * width, temp[2] * height)
                boxes.append(Box(lower,upper))
            i += 1
        return boxes

    def getBox(self, objective):
        tempTiles = getTiles(
            cv2.resize(
                self.image, 
                (790, 445), #enjoy the magic numbers
                interpolation = cv2.INTER_AREA)
        )
        x = 230
        y = 105
        for j in range(len(tempTiles)):
            tempTiles[j].lower.x += x
            tempTiles[j].upper.x += x
            tempTiles[j].lower.y += y
            tempTiles[j].upper.y += y

        MIN_DISPARITY = 300
        i = 0
        for tile in self.tiles:
            outputDict = self.od.run_inference_for_single_image(tile.tile)
            boxes = self.extractBoxes(outputDict, objective)
            for box in boxes:
                box.addOffset(tempTiles[i])
                if objective == DROP_BOX:
                    return(box, self.getDepthAtPoint(self.depth_image, box.getMiddle()))
                if self.getBoxDisparity(box) > MIN_DISPARITY and objective == DATE:
                    print("detected date!")
                    return (box, self.getboxMinDistance(box))
            i += 1
                
        return (Box(Point(0,0), Point(0,0)), 0)

    def get2DObjectiveCallback(self, req):
        self.tiles = getTiles(self.image)
        box, distance = self.getBox(req.objective)
        point = box.getMiddle()
        middle = 1280/2
        dpp = 86 / 1280
        return get2DObjectiveResponse((point.x - middle) * dpp, distance)
        

    
    def saveImagesCallback(self, objective):
        self.tileImages()

        cv2.imwrite('/home/alec/Pictures/cv_images/' + str(time.time_ns()) + '_uncut.jpg', self.image)
        cv2.imwrite('/home/alec/Pictures/cv_images/' + str(time.time_ns()) + '_1.jpg', self.tile1)
        cv2.imwrite('/home/alec/Pictures/cv_images/' + str(time.time_ns()) + '_2.jpg', self.tile2)
        cv2.imwrite('/home/alec/Pictures/cv_images/' + str(time.time_ns()) + '_3.jpg', self.tile3)
        cv2.imwrite('/home/alec/Pictures/cv_images/' + str(time.time_ns()) + '_4.jpg', self.tile4)
        

def main():
        rospy.init_node('get2DObjectiveServer')

        detector = Detector()
        rospy.Subscriber('camera/image', Image, detector.getColourImageCallback)
        rospy.Subscriber('camera/depth/image_rect_raw', Image, detector.getDepthImageCallback)
        rospy.Service('get2DObjective', get2DObjective, detector.get2DObjectiveCallback)
        rospy.Service('saveImages', saveImages, detector.saveImagesCallback)
        
        print("ready")
        rospy.spin()

if __name__ == '__main__':
    main()