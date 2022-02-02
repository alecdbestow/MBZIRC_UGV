#!/usr/bin/env python3

import numpy as np
import rospy
import cv2
import time

from ugv.srv import align
from cv_bridge import CvBridge
from sensor_msgs.msg import Image


class Aligner:
    def __init__(self) -> None:
        self.inputs = [105, 550, 230, 1020]

    def realsenseCallback(self, image_message):
        bridge = CvBridge()
        cv_image = bridge.imgmsg_to_cv2(image_message, desired_encoding='rgb8')
        self.realsense_image = np.asarray_chkfinite(cv_image)

    def cameraCallback(self, image_message):
        bridge = CvBridge()
        cv_image = bridge.imgmsg_to_cv2(image_message, desired_encoding='rgb8')
        self.camera_image = np.asarray_chkfinite(cv_image)
    
    def alignerCallback(self, temp):
        r_height = self.realsense_image.shape[0]
        r_width = self.realsense_image.shape[1]

        c_width = self.camera_image.shape[1]
        c_height = self.camera_image.shape[0]

        
        print(f"r_height: {r_height}, r_width: {r_width}, c_width: {c_width}, c_height: {c_height}, ")
        for i in range(4):
            data = input()
            if data:
                self.inputs[i] = int(data)
            

        adjusted_image = self.realsense_image[self.inputs[0]:self.inputs[1], self.inputs[2]:self.inputs[3]]
        adjusted_image = cv2.resize(adjusted_image, (c_width, c_height), interpolation = cv2.INTER_AREA)
        
        combined = cv2.addWeighted(self.camera_image, 0.4, adjusted_image, 0.4, 0)
        cv2.imwrite('/home/alec/Pictures/cv_images/' + str(time.time_ns()) + '_aligned.jpg', combined)

def main():
        rospy.init_node('imageAlignerServer')

        aligner = Aligner()
        rospy.Subscriber('camera/image', Image, aligner.cameraCallback)
        rospy.Subscriber('camera/infra2/image_rect_raw', Image, aligner.realsenseCallback)
        rospy.Service('align', align, aligner.alignerCallback)
        
        print("ready")
        rospy.spin()

if __name__ == '__main__':
    main()