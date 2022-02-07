#!/usr/bin/env python3

import rospy

def scanForObjective(objective):
    #scan
    return (0, 0)

def getDegrees(point):
    return 0

def adjustPosition(point, distance):
    print(f"rotate by: {getDegrees(point)} degrees")
    print(f"move forward by: {distance} mm")
    

def catchDate():
    point, distance = scanForObjective(DATE)
    
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
