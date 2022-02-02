#!/usr/bin/env python3

import debugpy

debugpy.listen(5678)
debugpy.wait_for_client()


import numpy as np
import os
import sys
import tarfile
import tensorflow as tf
import zipfile
import pathlib
import rospy
import PIL
import cv2
import time

from ugv.srv import get2DObjective, get2DObjectiveResponse, saveImages
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from IPython.display import display

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1

# Patch the location of gfile
tf.gfile = tf.io.gfile

class Detector:
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
        return output_dict


    def show_inference(self, image):
        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        # Actual detection.
        image_np = image.copy()
        #self.image.setflags(write=1)
        #image_np.setflags(write=1)
        output_dict = self.run_inference_for_single_image(image_np)

        left = 0
        right = 0
        upper = 0
        lower = 0

        depth_height = self.realsense_image.shape[0]
        depth_width = self.realsense_image.shape[1]

        colour_width = self.camera_image.shape[1]
        colour_height = self.camera_image.shape[0]

        OVERSIZE_RATIO = 0.1

        left = (depth_width/colour_width) * left
        right = (depth_width/colour_width) * right

        left = left - (right - left) * OVERSIZE_RATIO
        right = right + (right - left) * OVERSIZE_RATIO

        upper = upper - (lower - upper) * OVERSIZE_RATIO
        lower = lower 

        upper = (depth_height/colour_height) * upper
        lower = (depth_height/colour_height) * lower
        
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
        cv2.waitKey(0) # waits until a key is pressed
        


    def getColourImageCallback(self, image_message):
        bridge = CvBridge()
        cv_image = bridge.imgmsg_to_cv2(image_message, desired_encoding='bgr8')
        
        self.image = np.asarray_chkfinite(cv_image)

    def getDepthImageCallback(self, depth_message):
        if not self.depth_image:
            self.depth_image = Image()
        self.depth_image = depth_message
        self.getDepthAtPixel(self.depth_image, 100, 100)
        


    def getDepthAtPixel(self, depth_message, x, y):
        row_step = depth_message.step / depth_message.width
        pos = int(row_step * x + y * depth_message.step)
        return int.from_bytes(
            [depth_message.data[pos], depth_message.data[pos + 1]], 
            byteorder='little', 
            signed=False
        )

    def getBoxDisparity(self, p1, p2):
        max = 0
        min = 0
        for y in range(p1[1], p2[2]):
            for x in range(p1[0], p2[0]):
                dist = self.getDepthAtPixel(self.depth_image, x, y)
                if dist:
                    if dist < min:
                        min = dist
                    elif dist > max:
                        max = dist
        
        return max - min

    
    def get2DObjectiveCallback(self, objective):
        self.getDepthAtPixel(self.depth_image, 100, 100)
        '''
        print("geting objective")
        height = self.image.shape[0]
        width = self.image.shape[1]

        width_cutoff = width // 2
        height_cutoff = height // 2

        
        self.tile1 = self.image[height_cutoff:, :width_cutoff]
        self.tile2 = self.image[height_cutoff:, width_cutoff:]
        self.tile3 = self.image[:height_cutoff, :width_cutoff]
        self.tile4 = self.image[:height_cutoff, width_cutoff:]

        
        #self.tile3.show()
        print("after tiling")
        self.show_inference(self.tile1)
        self.show_inference(self.tile2)
        self.show_inference(self.tile3)
        self.show_inference(self.tile4)
        
        pass
    '''
    
    def saveImagesCallback(self, objective):
        height = self.image.shape[0]
        width = self.image.shape[1]

        width_cutoff = width // 2
        height_cutoff = height // 2

        
        tile1 = self.image[height_cutoff:, :width_cutoff]
        tile2 = self.image[height_cutoff:, width_cutoff:]
        tile3 = self.image[:height_cutoff, :width_cutoff]
        tile4 = self.image[:height_cutoff, width_cutoff:]

        cv2.imwrite('/home/alec/Pictures/cv_images/' + str(time.time_ns()) + '_uncut.jpg', self.image)
        cv2.imwrite('/home/alec/Pictures/cv_images/' + str(time.time_ns()) + '_1.jpg', tile1)
        cv2.imwrite('/home/alec/Pictures/cv_images/' + str(time.time_ns()) + '_2.jpg', tile2)
        cv2.imwrite('/home/alec/Pictures/cv_images/' + str(time.time_ns()) + '_3.jpg', tile3)
        cv2.imwrite('/home/alec/Pictures/cv_images/' + str(time.time_ns()) + '_4.jpg', tile4)
        

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