#!/usr/bin/env python3

import rospy
import time

from ugv.srv import get2DObjective, activateDropper
LEVEL = 130
DOWN = 165
DROP_BOX = 2
OPEN = 1
CLOSE = 0

class UGV:
    def __init__(self) -> None:
        rospy.init_node('main controller')
        rospy.wait_for_service('get2DObjective')
        rospy.wait_for_service('tilt')
        rospy.wait_for_service('dropDates')
        self.get2DObjective = rospy.ServiceProxy('get2DObjective', get2DObjective)
        self.dateService = rospy.ServiceProxy('dropDates', activateDropper)
        self.tilt = rospy.ServiceProxy('tilt', activateDropper)
        self.tilt(LEVEL)
        self.dateService(0)

    def scanForObjective(self, objective):
        resp = self.get2DObjective(objective)
        distance = resp.distance
        respRotate = resp.rotate
        if distance == 0:
            print("did not detect")
            self.rotate(20)
            resp = self.get2DObjective(objective)
            distance = resp.distance
            respRotate = resp.rotate
        
        if distance == 0:
            print("did not detect")
            self.rotate(-40)
            resp = self.get2DObjective(objective)
            distance = resp.distance
            respRotate = resp.rotate
        
        return (respRotate, distance)

    def rotate(self, degrees):
        if degrees != 0:
            print(f"rotate by: {degrees} degrees")
            print("Press any key when finished")
            input()
    
    def travel(self, distance):
        if distance != 0:
            print(f"move forward by: {distance} mm")
            print("Press any key when finished")
            input()
        


    def adjustPosition(self, degrees, distance):
        self.rotate(degrees)
        self.travel(distance)

    def catchDate(self):
        print("catching date")
        DATE = 1
        degrees, distance = self.scanForObjective(DATE)
        self.adjustPosition(degrees, distance)
        
        return distance != 0

    def dropDate(self):
        print("dropping date")
        self.tilt(DOWN)
        time.sleep(2)
        rotation, distance = self.scanForObjective(DROP_BOX)
        if distance == 0:
            print("ERROR CANNOT FIND DROP BOX!")
            self.adjustPosition(20, 2000)
            
        self.tilt(LEVEL)
        self.adjustPosition(rotation, distance)
        self.dateService(OPEN)
        time.sleep(2)
        self.dateService(CLOSE)
        time.sleep(2) 


def main():
    
    ugv = UGV()
    print("ready! press any key to start")
    input()
    while ugv.catchDate():
        ugv.rotate(180)
        ugv.dropDate()
        ugv.rotate(180)

if __name__ == '__main__':
    main()
