/*
 * rosserial Servo Control Example
 *
 * This sketch demonstrates the control of hobby R/C servos
 * using ROS and the arduiono
 * 
 * For the full tutorial write up, visit
 * www.ros.org/wiki/rosserial_arduino_demos
 *
 * For more information on the Arduino Servo Library
 * Checkout :
 * http://www.arduino.cc/en/Reference/Servo
 */

#if (ARDUINO >= 100)
 #include <Arduino.h>
#else
 #include <WProgram.h>
#endif



#include <Servo.h> 
#include <ros.h>
#include <std_msgs/UInt16.h>

ros::NodeHandle  nh;

Servo dateServo;
Servo tiltServo;

void servo_cb( const std_msgs::UInt16& cmd_msg){
  Serial.println(cmd_msg.data);
  dateServo.write(cmd_msg.data); //set servo angle, should be from 0-180  
  digitalWrite(13, HIGH-digitalRead(13));  //toggle led  
}

void tilt_cb(const std_msgs::UInt16& cmd_msg) {
  Serial.println(cmd_msg.data);
    tiltServo.write(cmd_msg.data); //set servo angle, should be from 0-180  
}


ros::Subscriber<std_msgs::UInt16> dateSub("DateServo", servo_cb);
ros::Subscriber<std_msgs::UInt16> tiltSub("TiltServo", tilt_cb);

void setup(){
  Serial.begin(57600);
  nh.initNode();
  nh.subscribe(dateSub);
  nh.subscribe(tiltSub);
  
  dateServo.attach(9); //attach it to pin 9
  tiltServo.attach(10);
}

void loop(){
  nh.spinOnce();
  delay(1);
}
