#!/usr/bin/env python3

import debugpy

debugpy.listen(5678)

debugpy.wait_for_client()


import numpy as np
import rospy
import cv2

from ugv.srv import get2DObjective, get2DObjectiveResponse, saveImages
from ugv.msg import target
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from IPython.display import display

DATE = 1
DROP_BOX = 2

class ObjectDetector:
    def __init__(self):
        pass

    def run_inference_for_single_image(self, image):

        frame = image.copy()
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
        low_red = np.array([161, 155, 84])  
        high_red = np.array([179, 255, 255])
        red_mask = cv2.inRange(hsv_frame, low_red, high_red)
        red = cv2.bitwise_and(frame, frame, mask=red_mask)
        
        low_green = np.array([25, 52, 150])
        high_green = np.array([102, 255, 255])
        green_mask = cv2.inRange(hsv_frame, low_green, high_green)
        green = cv2.bitwise_and(frame, frame, mask=green_mask)


        # Setup SimpleBlobDetector parameters.
        params = cv2.SimpleBlobDetector_Params()

        # Change thresholds
        params.minThreshold = 10
        params.maxThreshold = 200

        # Filter by Area.
        params.filterByArea = True
        params.minArea = 1500

        # Detect blobs.


        # Create a detector with the parameters
        ver = (cv2.__version__).split('.')
        if int(ver[0]) < 3 :
            detector = cv2.SimpleBlobDetector(params)
        else : 
            detector = cv2.SimpleBlobDetector_create(params)

        keypoints = detector.detect(red_mask)

        # Draw detected blobs as red circles.
        # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
        im_with_keypoints = cv2.drawKeypoints(red_mask, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        # Show keypoints
        cv2.imshow("Keypoints", im_with_keypoints)
        cv2.waitKey(0)

        return 0

    def show_inference(self, image):
        pass
        # the array based representation of the image will be used later in order to prepare the
    
 

class Point:
    def __init__(self, X, Y):
        self.x = X
        self.y = Y

    def __add__(self, p1):
        return Point(self.x + p1.x, self.y + p1.y)
    
    def __truediv__(self, div):
        return Point(self.x / div, self.y / div)


class Detector:
    def __init__(self):
        self.od = ObjectDetector()

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


    def get2DObjectiveCallback(self, req):
        self.od.run_inference_for_single_image(self.image)
        point = Point(0,0)
        middle = 1280/2
        dpp = 86 / 1280
        return get2DObjectiveResponse((point.x - middle) * dpp, distance)
        

    
        

def main():
        rospy.init_node('get2DObjectiveServer')

        detector = Detector()
        rospy.Subscriber('camera/image', Image, detector.getColourImageCallback)
        rospy.Subscriber('camera/depth/image_rect_raw', Image, detector.getDepthImageCallback)
        rospy.Service('get2DObjective', get2DObjective, detector.get2DObjectiveCallback)
        
        print("ready")
        rospy.spin()

if __name__ == '__main__':
    main()