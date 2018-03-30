#include <ros/ros.h>
#include <math.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>
#include <dynamic_reconfigure/server.h>
#include <blimp_navigation/ControllerConfig.h>
#include <string>
#include <vector>
#include <algorithm>
#include <std_msgs/Float64.h>
#include <std_msgs/Int16.h>


const double PI = 3.141592653589793;

class BlimpController {
    private:
        ros::NodeHandle nh_;
        dynamic_reconfigure::Server<blimp_navigation::ControllerConfig> server;
        dynamic_reconfigure::Server<blimp_navigation::ControllerConfig>::CallbackType f;
        tf::TransformListener listener;
        tf::TransformBroadcaster br;
        ros::Publisher dist_pub;
        ros::Publisher dist_y_pub;
        ros::Publisher z_pub;
        ros::Publisher cmd_yaw_pub;
        ros::Subscriber yaw_sub;
        tf::Transform transform;
        tf::Vector3 goal;
        tf::Quaternion q;
        short int state;        // 0 = Stop; 1 = Turn; 2 = Move
        float prev_dist;
        float prev_z;
        float prev_angle;
        float yaw_blimp;
        float command_yaw;
        float goal_vicinity;
        float stop_criteria_rotate;
        float stop_criteria_move;
        int count_stop;
        
        bool inGoalVicinity(tf::Vector3 distVec) {
            return sqrt(pow(distVec.x(),2) + pow(distVec.y(),2)) <= goal_vicinity;
        }
        
        bool stoppedMoving(tf::Vector3 distVect) {
            if (fabs(distVect.x() - prev_dist) <= stop_criteria_move)
                count_stop++;
            else
                count_stop = 0;
            return (count_stop >= 10);
        }
        
        bool stoppedRotating() {
            if (fabs(yaw_blimp - prev_angle) <= stop_criteria_rotate)
                count_stop++;
            else
                count_stop = 0;
            return (count_stop >= 10);
        }
        
    public:
        BlimpController()
        {
            f = boost::bind(&BlimpController::callback, this, _1, _2);
            server.setCallback(f);
            dist_pub = nh_.advertise<std_msgs::Float64>("dist",1);
            dist_y_pub = nh_.advertise<std_msgs::Float64>("dist_y",1);
            z_pub = nh_.advertise<std_msgs::Float64>("z",1);
            cmd_yaw_pub = nh_.advertise<std_msgs::Int16>("cmd_yaw",1);
            yaw_sub = nh_.subscribe("yaw", 1, &BlimpController::yawCallback, this);
            goal = tf::Vector3(0.0,0.0,0.0);
            q.setRPY(0.0, 0.0, 0.0);
            transform.setOrigin(goal);
            transform.setRotation(q);
            state = 0;
            ROS_INFO("%d", state);
            prev_dist = 1e6;
            prev_z = 1e6;
            prev_angle = -2*PI;
            yaw_blimp = 0.;
            goal_vicinity = 0.3;        // 30 cm
            stop_criteria_rotate = 1.*PI/180.;
            stop_criteria_move = 0.0005;      // 1 cm/s at 10 Hz
        }
        
        void callback(blimp_navigation::ControllerConfig &config, uint32_t level){
            if (config.groups.goal.set) {
                goal = tf::Vector3(config.groups.goal.x, config.groups.goal.y, config.groups.goal.z);
                q.setRPY(0.0, 0.0, config.groups.goal.yaw*PI/180);
                transform.setOrigin(goal);
                transform.setRotation(q);
            }
        }
        
        void yawCallback(const std_msgs::Int16::ConstPtr& msg) {
            yaw_blimp = float(msg->data) * PI/180.;
            std::vector<std::string> frames;
            tf::StampedTransform rotation;
            ros::Time last_time;
            std::string *error;
            try {
                listener.getFrameStrings(frames);
                if (std::find(frames.begin(), frames.end(), "blimp") != frames.end()) {
                    tf::StampedTransform blimp_tf;
                    listener.lookupTransform("world", "blimp", ros::Time(0), blimp_tf);
                    q.setRPY(0.0, 0.0, yaw_blimp);
                    rotation.setOrigin(tf::Vector3(0.,0.,0.));
                    rotation.setRotation(q);
                    br.sendTransform(tf::StampedTransform(rotation, ros::Time::now(), "blimp", "blimp_rotated"));
                }
            }
            catch (tf::TransformException ex) {
                ROS_ERROR("s", ex.what());
            }
        }
        
        void publish(){
            br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "world", "goal"));
        }
        
        void navigate(){
            std::vector<std::string> frames;
            tf::StampedTransform transform;
            ros::Time last_time;
            std::string *error;
            try{
                listener.getFrameStrings(frames);
                if (std::find(frames.begin(), frames.end(), "blimp_rotated") != frames.end()){
                    listener.getLatestCommonTime("blimp_rotated","goal",last_time,error);
                    if (ros::Time::now()-last_time > ros::Duration(1))
                    {
                        prev_dist = 1e6;
                        prev_z = 1e6;
                        prev_angle = -2*PI;
                        return;         // Too old
                    }
                    //listener.lookupTransform("blimp", "goal", ros::Time(0), transform);
                    tf::StampedTransform t_blimp;
                    listener.lookupTransform("blimp_rotated","goal", last_time, t_blimp);
                    tf::Vector3 d_blimp = t_blimp.getOrigin();
                    
                    switch (state) {
                        case 0:         // Stop
                            // Check transition condition
                            
                            if (!inGoalVicinity(d_blimp)) {
                                if (fabs(d_blimp.y()) > 0.1) {     // If going direct --> > 10 cm lateral errorr --> need angle adjustment --> turn
                                    double y_turn = atan2(d_blimp.y(), d_blimp.x());
                                    /*if (fabs(y_turn) > PI/2){
                                        command_yaw = yaw_blimp + y_turn - PI;        // Reversed
                                    }else{
                                        command_yaw = yaw_blimp + y_turn;
                                    }*/
                                    command_yaw = yaw_blimp + y_turn;
                                    if (command_yaw > PI)
                                        command_yaw -= 2*PI;
                                    else if (command_yaw <= -PI)
                                        command_yaw += 2*PI;
                                        
                                    std_msgs::Int16 cmd;
                                    cmd.data = round(command_yaw*180./PI);
                                    cmd_yaw_pub.publish(cmd);
                                    state = 1;
                                    ROS_INFO("%d", state);
                                }
                                else {
                                    state = 2;
                                    ROS_INFO("%d", state);
                                }
                            }
                            else {
                                tf::Matrix3x3 m2(t_blimp.getRotation());
                                double r_turn, p_turn, y_turn;
                                m2.getRPY(r_turn, p_turn, y_turn);
                                if (fabs(y_turn) > 0.1) {
                                    command_yaw = yaw_blimp + y_turn;
                                    if (command_yaw > PI)
                                        command_yaw -= 2*PI;
                                    else if (command_yaw <= -PI)
                                        command_yaw += 2*PI;
                                        
                                    ROS_INFO("command_yaw changed to %f", command_yaw);
                                    std_msgs::Int16 cmd;
                                    cmd.data = round(command_yaw*180./PI);
                                    cmd_yaw_pub.publish(cmd);
                                    state = 1;
                                    ROS_INFO("%d", state);
                                }
                                else {
                                    // Hovering algorithm
                                    std_msgs::Float64 dist_msg;
                                    dist_msg.data = d_blimp.x();
                                    dist_pub.publish(dist_msg);
                                    std_msgs::Float64 dist_y_msg;
                                    dist_y_msg.data = d_blimp.y();
                                    dist_y_pub.publish(dist_y_msg);
                                }
                            }
                            
                            break;
                                
                        case 1:         // Turn
                            // Check transition condition
                            if (fabs(yaw_blimp - command_yaw) <= 0.1 && stoppedRotating()) {
                                // Good alignment and already at low turning speed --> Stop
                                count_stop = 0;
                                state = 0;
                                ROS_INFO("%d", state);
                            }
                            else {
                                double r_turn, p_turn, y_turn;
                                // Need to adjust?
                                if (inGoalVicinity(d_blimp)) {
                                    tf::Matrix3x3 m2(t_blimp.getRotation());
                                    m2.getRPY(r_turn, p_turn, y_turn);
                                }
                                else {
                                    y_turn = atan2(d_blimp.y(), d_blimp.x());
                                }
                                float new_cmd = yaw_blimp + y_turn;
                                if (new_cmd > PI)
                                    new_cmd -= 2*PI;
                                else if (new_cmd <= -PI)
                                    new_cmd += 2*PI;
                                    
                                if (fabs(new_cmd - command_yaw) > 0.1)  {    // difference of the command to the current situation (the blimp may drift and need to turn more/less to the goal)
                                    // Set new command and send
                                    command_yaw = new_cmd;
                                    ROS_INFO("command_yaw changed to %f", command_yaw);
                                    std_msgs::Int16 cmd;
                                    cmd.data = round(command_yaw*180./PI);
                                    cmd_yaw_pub.publish(cmd);
                                }
                                else {
                                    // Keep turning
                                }
                            }
                            
                            break;
                            
                        case 2:         // Move
                            // Check transition condition
                            if (inGoalVicinity(d_blimp) && stoppedMoving(d_blimp)) {
                                // Close and stop --> Stop
                                count_stop = 0;
                                state = 0;
                                ROS_INFO("%d", state);
                            }
                            else {
                                std_msgs::Float64 dist_y_msg;
                                dist_y_msg.data = d_blimp.y();
                                dist_y_pub.publish(dist_y_msg);
                                
                                // Moving algorithm
                                std_msgs::Float64 dist_msg;
                                dist_msg.data = d_blimp.x();
                                dist_pub.publish(dist_msg);
                            }
                            break;
                    }
                    // Altitude control
                    std_msgs::Float64 z_msg;
                    z_msg.data = d_blimp.z();
                    z_pub.publish(z_msg);
                    
                    prev_dist = d_blimp.x();
                    prev_z = d_blimp.z();
                    prev_angle = yaw_blimp;
                    
                    
                }
            }
            catch (tf::TransformException ex){
                ROS_ERROR("%s", ex.what());
            }
        }
};


int main (int argc, char **argv) {
    ros::init(argc, argv, "blimp_controller");//, ros::init_options::AnonymousName);

    BlimpController blimp_controller;
    ros::Rate rate(10);
    while (ros::ok()){
        blimp_controller.publish();
        blimp_controller.navigate();
        ros::spinOnce();
        rate.sleep();
    }
    return 0;
}
