#!/usr/bin/env python3

import rospy

from ugv.srv import get2DObjective

class UGV:
    def __init__(self) -> None:
        rospy.wait_for_service('get2DObjective')
        self.get2DObjective = rospy.ServiceProxy('get2DObjective', get2DObjective)

    def scanForObjective(self, objective):
        point, distance = self.get2DObjective(objective)
        if distance == 0:
            print("rotate by: 20 degrees")
        
        return (0, 0)

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
        

    def getDegrees(point):
        return 0

    def adjustPosition(self, point, distance):
        self.rotate(self.getDegrees(point))
        self.travel(distance)

    def catchDate(searching = False):
        point, distance = scanForObjective(DATE)
        searching
        return distance != 0

    def dropDate():
        pass


def main():
    rospy.init_node('main controller')
    while catchDate():
        print("rotate by: 180 degrees")
        dropDate()
        print("rotate by: 180 degrees")

if __name__ == '__main__':
    main()
