#!/usr/bin/env python3

import rospy
import time

from ugv.srv import get2DObjective, activateDropper


class UGV:
    def __init__(self) -> None:
        rospy.init_node('main controller')
        rospy.wait_for_service('get2DObjective')
        self.get2DObjective = rospy.ServiceProxy('get2DObjective', get2DObjective)
        self.dateService = rospy.ServiceProxy('dropDates', activateDropper)

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
        

    def getDegrees(self, point):
        return 0

    def adjustPosition(self, point, distance):
        self.rotate(self.getDegrees(point))
        self.travel(distance)

    def catchDate(self):
        DATE = 1
        point, distance = self.scanForObjective(DATE)
        self.adjustPosition(point, distance)
        
        return distance != 0

    def dropDate(self):
        DROP_BOX = 2
        OPEN = 1
        CLOSE = 0
        point, distance = self.scanForObjective(DROP_BOX)
        if distance == 0:
            print("ERROR CANNOT FIND DROP BOX!")
            input()
        self.adjustPosition(point, distance)
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
