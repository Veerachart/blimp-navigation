/*
 * visual_servo_node.cpp
 *
 *  Created on: Jun 16, 2018
 *      Author: veerachart
 */

#include <ros/ros.h>
#include <gpu_opencv/FaceServo.h>
#include <std_msgs/Float64.h>
#include <std_msgs/Int16.h>

using namespace std;

class VisualServo {
protected:
    ros::Publisher err_y_pub_;
    ros::Publisher err_size_pub_;
    ros::Publisher left_right_pub_;
    ros::Publisher yaw_pub_;

    std_msgs::Float64 msg_y, msg_size, msg_lr;
    std_msgs::Int16 msg_yaw;

    ros::Subscriber face_sub_;
    ros::Subscriber yaw_sub_;

    ros::Time last_time;

    short current_yaw;

public:
    VisualServo();
    void faceCallback(const gpu_opencv::FaceServo::ConstPtr &msg);
    void yawCallback(const std_msgs::Int16::ConstPtr &msg);
    void run();
};

VisualServo::VisualServo() {
    ros::NodeHandle nh;
    face_sub_ = nh.subscribe("face", 1, &VisualServo::faceCallback, this);
    yaw_sub_ = nh.subscribe("yaw", 1, &VisualServo::yawCallback, this);

    err_y_pub_ = nh.advertise<std_msgs::Float64>("err_y", 1);
    err_size_pub_ = nh.advertise<std_msgs::Float64>("err_size", 1);
    left_right_pub_ = nh.advertise<std_msgs::Float64>("left_right", 1);
    yaw_pub_ = nh.advertise<std_msgs::Int16>("cmd_yaw", 1);

    last_time = ros::Time::now();

    current_yaw = 0;
}

void VisualServo::faceCallback(const gpu_opencv::FaceServo::ConstPtr &msg) {
    msg_y.data = (double) msg->y;
    err_y_pub_.publish(msg_y);
    msg_size.data = (double) msg->size;
    err_size_pub_.publish(msg_size);

    int direction = 0;      // 0: front, 1: left, -1:right
    float best = msg->prob_f;
    if (msg->prob_l > best) {
        direction = 1;
        best = msg->prob_l;
    }
    if (msg->prob_r > best) {
        direction = -1;
    }
    msg_lr.data = (double) direction;
    left_right_pub_.publish(msg_lr);

    short yaw = short(msg->x / 60.) + current_yaw;        // *60 degree / 360 px for half of the width
    while (yaw > 180)
        yaw -= 360;
    while(yaw <= -180)
        yaw += 360;
    msg_yaw.data = yaw;
    yaw_pub_.publish(msg_yaw);

    last_time = msg->header.stamp;
}

void VisualServo::yawCallback(const std_msgs::Int16::ConstPtr &msg) {
    current_yaw = msg->data;
}

void VisualServo::run() {
    if (ros::Time::now() - last_time > ros::Duration(2.0)) {
        // not detected more than 2 seconds
        msg_y.data = 0.;
        err_y_pub_.publish(msg_y);
        msg_size.data = 50.;
        err_size_pub_.publish(msg_size);
        msg_lr.data = 0.;
        left_right_pub_.publish(msg_lr);
        msg_yaw.data = current_yaw;
        yaw_pub_.publish(msg_yaw);
        last_time = ros::Time::now();
    }
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "visual_servo_node");
    VisualServo servoNode;
    ros::Rate r(30);

    while (ros::ok()) {
        servoNode.run();
        ros::spinOnce();
        r.sleep();
    }
    //ros::spin();

    return 0;
}
