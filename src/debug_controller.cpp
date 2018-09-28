/*
 * debug_controller.cpp
 *
 *  Created on: Aug 13, 2018
 *      Author: veerachart
 */
#include <ros/ros.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>
#include <string>
#include <vector>

const double PI = 3.141592653589793;

bool inGoalVicinity(tf::Vector3 distVec) {
    return sqrt(pow(distVec.x(),2) + pow(distVec.y(),2)) <= 0.5;
}

int main (int argc, char **argv) {
    ros::init(argc, argv, "debug_controller");//, ros::init_options::AnonymousName);
    tf::TransformListener listener;
    tf::TransformBroadcaster br;
    ros::Rate rate(10);
    ros::Time last_time;
    std::string *error;
    float command_yaw = 1.981127;
    float yaw_blimp = -2.967060;
    double y_turn;
    while (ros::ok()){
        try{
            if (listener.frameExists("blimp")){
                listener.getLatestCommonTime("blimp","goal",last_time,error);
                //listener.lookupTransform("blimp", "goal", ros::Time(0), transform);
                tf::StampedTransform t_blimp;
                listener.lookupTransform("blimp","goal", last_time, t_blimp);
                tf::StampedTransform blimp_world;
                listener.lookupTransform("world","blimp", ros::Time(0), blimp_world);
                tf::Vector3 d_blimp = t_blimp.getOrigin();
                if (inGoalVicinity(d_blimp)) {
                    y_turn = command_yaw - yaw_blimp;
                    ROS_INFO("1: %f", y_turn);
                }
                else {
                    y_turn = atan2(d_blimp.y(), d_blimp.x());
                    ROS_INFO("2: %f", y_turn);
                }
                ROS_INFO("goal_changed reset");

                float new_cmd = yaw_blimp + y_turn;
                if (new_cmd > PI)
                    new_cmd -= 2*PI;
                else if (new_cmd <= -PI)
                    new_cmd += 2*PI;
                if (fabsf(new_cmd - command_yaw) > PI/18. && fabsf(new_cmd - command_yaw) < 35.*PI/18.)  {    // difference of the command to the current situation (the blimp may drift and need to turn more/less to the goal)
                    // Set new command and send
                    command_yaw = new_cmd;
                    ROS_INFO("command_yaw changed to %f", command_yaw*180./PI);
                }
                else {
                    ROS_INFO("command_yaw not changed. new_cmd = %f, command_yaw = %f", new_cmd, command_yaw);
                    ROS_INFO("Diff: %f", new_cmd - command_yaw);
                    ROS_INFO("Abs diff: %f", fabsf(new_cmd - command_yaw));
                }
            }
        }
        catch (tf::TransformException &ex){
            ROS_ERROR("%s", ex.what());
        }
        rate.sleep();

    }
}
