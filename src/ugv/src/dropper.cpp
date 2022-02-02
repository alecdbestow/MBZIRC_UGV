#include "ros/ros.h"
#include "ugv/activateDropper.h"
#include <std_msgs/UInt16.h>

constexpr int CLOSED = 180;
constexpr int OPEN = 0;

class Dropper   {
    public:
        Dropper();
        bool service(ugv::activateDropper::Request &req, ugv::activateDropper::Response &res);
        std_msgs::UInt16 angle;
        
};

Dropper::Dropper()  {
    angle.data = CLOSED;
}
bool Dropper::service(ugv::activateDropper::Request &req, ugv::activateDropper::Response &res)
{
    if (req.status == 0) {
        angle.data = CLOSED;
    }   else if (req.status == 1)    {
        angle.data = OPEN;
    }   else if (req.status == 2 && angle.data == OPEN) {
        angle.data = CLOSED;
    }   else    {
        angle.data = OPEN;
    }
        
    res.worked = true;
    return true;
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "add_two_ints_server");

    ros::NodeHandle nh;
    ros::Publisher publisher = nh.advertise<std_msgs::UInt16>("servo",1);
    Dropper dropper;
    // start the dropper in the open position
    ros::Rate loop_rate(10);
    int count = 0;
    ros::ServiceServer service = nh.advertiseService("activateDropper", &Dropper::service, &dropper);

    while (ros::ok())   {
        publisher.publish(dropper.angle);
        ros::spinOnce();
        loop_rate.sleep();
        ++count;
    }
    
    
    ROS_INFO("Ready to activate dropper");
    ros::spin();

    return 0;
}