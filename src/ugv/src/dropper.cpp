#include "ros/ros.h"
#include "ugv/activateDropper.h"
#include <std_msgs/UInt16.h>

constexpr int CLOSED = 180;
constexpr int OPEN = 0;
constexpr int LEVEL = 135;

class Dropper   {
    public:
        Dropper();
        bool dropDates(ugv::activateDropper::Request &req, ugv::activateDropper::Response &res);
        bool tilt(ugv::activateDropper::Request &req, ugv::activateDropper::Response &res);
        std_msgs::UInt16 dateAngle;
        std_msgs::UInt16 tiltAngle;
        
};

Dropper::Dropper()  {
    dateAngle.data = CLOSED;
    tiltAngle.data = LEVEL;
}
bool Dropper::dropDates(ugv::activateDropper::Request &req, ugv::activateDropper::Response &res)
{
    if (req.status == 0) {
        dateAngle.data = CLOSED;
    }   else if (req.status == 1)    {
        dateAngle.data = OPEN;
    }   else if (req.status == 2 && dateAngle.data == OPEN) {
        dateAngle.data = CLOSED;
    }   else    {
        dateAngle.data = OPEN;
    }
        
    res.worked = true;
    return true;
}

bool Dropper::tilt(ugv::activateDropper::Request &req, ugv::activateDropper::Response &res)
{
    tiltAngle.data = req.status;
    res.worked = true;
    return true;
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "add_two_ints_server");

    ros::NodeHandle nh;
    ros::Publisher datePublisher = nh.advertise<std_msgs::UInt16>("DateServo",1);
    ros::Publisher tiltPublisher = nh.advertise<std_msgs::UInt16>("TiltServo",1);
    Dropper dropper;
    // start the dropper in the open position
    ros::Rate loop_rate(100);
    int count = 0;
    ros::ServiceServer dateService = nh.advertiseService("dropDates", &Dropper::dropDates, &dropper);
    ros::ServiceServer tilt = nh.advertiseService("tilt", &Dropper::tilt, &dropper);

    std_msgs::UInt16 oldTilt = 0;
    std_msgs::UInt16 oldDate = 0;

    while (ros::ok())   {

        if (dropper.dateAngle != oldDate)    {
            datePublisher.publish(dropper.dateAngle);
            oldDate = dropper.dateAngle;
            
        }
        if (dropper.tiltAngle != oldTilt)   {
            tiltPublisher.publish(dropper.tiltAngle);
            oldTilt = dropper.tiltAngle;
        }
        
        
        
        ros::spinOnce();
        loop_rate.sleep();
        ++count;
    }
    
    
    ROS_INFO("Ready to activate dropper");
    ros::spin();

    return 0;
}