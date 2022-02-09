#include "ros/ros.h"
#include "ugv/activateDropper.h"
#include <std_msgs/UInt16.h>

constexpr int CLOSED = 180;
constexpr int OPEN = 0;

class Dropper   {
    public:
        Dropper();
        bool dropDates(ugv::activateDropper::Request &req, ugv::activateDropper::Response &res);
        bool dropBlanket(ugv::activateDropper::Request &req, ugv::activateDropper::Response &res);
        std_msgs::UInt16 dateAngle;
        std_msgs::UInt16 blanketAngle;
        
};

Dropper::Dropper()  {
    dateAngle.data = CLOSED;
    blanketAngle.data = CLOSED;
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

bool Dropper::dropBlanket(ugv::activateDropper::Request &req, ugv::activateDropper::Response &res)
{
    if (req.status == 0) {
        blanketAngle.data = CLOSED;
    }   else if (req.status == 1)    {
        blanketAngle.data = OPEN;
    }   else if (req.status == 2 && blanketAngle.data == OPEN) {
        blanketAngle.data = CLOSED;
    }   else    {
        blanketAngle.data = OPEN;
    }
        
    res.worked = true;
    return true;
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "add_two_ints_server");

    ros::NodeHandle nh;
    ros::Publisher datePublisher = nh.advertise<std_msgs::UInt16>("DateServo",1);
    ros::Publisher blanketPublisher = nh.advertise<std_msgs::UInt16>("BlanketServo",1);
    Dropper dropper;
    // start the dropper in the open position
    ros::Rate loop_rate(10);
    int count = 0;
    ros::ServiceServer dateService = nh.advertiseService("dropDates", &Dropper::dropDates, &dropper);
    ros::ServiceServer blanketService = nh.advertiseService("dropBlanket", &Dropper::dropBlanket, &dropper);

    while (ros::ok())   {
        datePublisher.publish(dropper.dateAngle);
        blanketPublisher.publish(dropper.blanketAngle);
        ros::spinOnce();
        loop_rate.sleep();
        ++count;
    }
    
    
    ROS_INFO("Ready to activate dropper");
    ros::spin();

    return 0;
}