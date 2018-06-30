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
#include <std_msgs/Bool.h>
#include <fstream>


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
        ros::Publisher reset_x_pub;
        ros::Publisher reset_y_pub;
        ros::Subscriber yaw_sub;
        tf::Transform transform;
        tf::Transform pinpoint_tf;       // For keeping the location of the blimp prior to the first rotation
        tf::Vector3 goal;
        tf::Vector3 goal_human;
        tf::Vector3 human_avg;
        float human_yaw_avg;
        tf::Quaternion q;
        tf::Quaternion q_goal;
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
        float distance_to_goal;
        float distance_border;
        float angle_border;
        int count_lock;             // counter for averaging position & lock the goal position/orientation
        int count_limit;            // limit for count_lock so the average stops when count_limit times of human position has been averaged
        bool use_human_control;     // flag to use human position and direction to set the goal of the blimp
        bool goal_locked;           // flag that the goal is now locked and fixed for navigation
        bool to_navigate;           // used to start the navigation (so that we don't start driving from the beginning)
        
        std::ofstream file;          // file for saving goal position

        std_msgs::Bool reset_msg;

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
        
        void clearDrivePowers() {
            reset_x_pub.publish(reset_msg);
            reset_y_pub.publish(reset_msg);
        }

    public:
        BlimpController(std::string const &file_name) : file(file_name.c_str(), std::ios::out)
        {
            f = boost::bind(&BlimpController::callback, this, _1, _2);
            server.setCallback(f);
            dist_pub = nh_.advertise<std_msgs::Float64>("dist",1);
            dist_y_pub = nh_.advertise<std_msgs::Float64>("dist_y",1);
            z_pub = nh_.advertise<std_msgs::Float64>("z",1);
            cmd_yaw_pub = nh_.advertise<std_msgs::Int16>("cmd_yaw",1);
            reset_x_pub = nh_.advertise<std_msgs::Bool>("reset_x",1);
            reset_y_pub = nh_.advertise<std_msgs::Bool>("reset_y",1);
            reset_msg.data = true;
            yaw_sub = nh_.subscribe("yaw", 1, &BlimpController::yawCallback, this);
            goal = tf::Vector3(0.0,0.0,0.0);
            goal_human = tf::Vector3(0.0,0.0,0.0);
            q.setRPY(0.0, 0.0, 0.0);
            transform.setOrigin(goal);
            transform.setRotation(q);
            state = 0;
            ROS_INFO("%d", state);
            prev_dist = 1e6;
            prev_z = 1e6;
            prev_angle = -2*PI;
            yaw_blimp = 0.;
            goal_vicinity = 0.4;        // 40 cm
            stop_criteria_rotate = 2.*PI/180.;
            stop_criteria_move = 0.005;      // 5 cm/s at 10 Hz
            distance_to_goal = 1.0;             // 1 m
            distance_border = 0.15;             // 15 cm (from ROBIO)
            angle_border = 0.2618;              // 15 degree
            count_limit = 5;                    // half a second at 10 Hz first
            goal_locked = false;
            to_navigate = false;

            file << "Time,goal_x,goal_y,goal_z,goal_yaw" << std::endl;
            file << std::fixed << std::setprecision(6);
        }
        
        void callback(blimp_navigation::ControllerConfig &config, uint32_t level){
            to_navigate = config.groups.goal.navigate;
            if (config.groups.goal.set) {
                goal = tf::Vector3(config.groups.goal.x, config.groups.goal.y, config.groups.goal.z);
                q.setRPY(0.0, 0.0, config.groups.goal.yaw*PI/180);
                transform.setOrigin(goal);
                transform.setRotation(q);
                return;
            }


            if (config.groups.goal.human) {
                use_human_control = true;
            }
            else {
                use_human_control = false;
                goal_locked = false;
            }
        }
        
        void yawCallback(const std_msgs::Int16::ConstPtr& msg) {
            yaw_blimp = -float(msg->data) * PI/180.;
        }
        
        void publish(){
            br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "world", "goal"));
            if (state == 1)
                br.sendTransform(tf::StampedTransform(pinpoint_tf, ros::Time::now(), "world", "pinpoint"));
            file << ros::Time::now().toSec() << "," << transform.getOrigin().x() << "," << transform.getOrigin().y() << "," << transform.getOrigin().z() << "," << tf::getYaw(q)*180./PI << std::endl;
            if (use_human_control) {
                std::vector<std::string> frames;
                ros::Time last_time;
                std::string *error;
                try {
                    listener.getFrameStrings(frames);
                    if (std::find(frames.begin(), frames.end(), "human0") != frames.end()) {
                        tf::StampedTransform human_tf;
                        listener.getLatestCommonTime("world", "human0", last_time, error);
                        if (ros::Time::now() - last_time > ros::Duration(1.0)) {
                            ROS_ERROR("Head too old");
                            return;
                        }
                        listener.lookupTransform("world", "human0", last_time, human_tf);
                        tf::Vector3 human = human_tf.getOrigin();
                        tf::Quaternion q_human = human_tf.getRotation();
                        float yaw = tf::getYaw(q_human);
                        if (goal_locked) {
                            tf::StampedTransform check_tf;
                            ros::Time common_time;
                            listener.getLatestCommonTime("goal", "human0", common_time, error);
                            if (ros::Time::now() - common_time > ros::Duration(1.0)) {
                                ROS_ERROR("Head & goal not close enough in time");
                                goal_locked = false;
                                return;
                            }
                            listener.lookupTransform("goal", "human0", common_time, check_tf);
                            ROS_INFO("%.3f, %.3f, %.3f --- %.3f", check_tf.getOrigin().x(), check_tf.getOrigin().y(), check_tf.getOrigin().z(), 180./PI*tf::getYaw(check_tf.getRotation()));
                        }
                        else {
                            if (count_lock < count_limit) {
                                // accumulating
                                human_avg += human;
                                if (count_lock == 0) {
                                    // First one, no need to care about the sign
                                    human_yaw_avg += yaw;
                                }
                                else {
                                    // From the second, need to keep it on the same side as the first (wrapping problem)
                                    if (yaw > PI/2. || yaw <= -PI/2.) {
                                        human_yaw_avg += yaw;           // TODO
                                    }
                                    else {
                                        human_yaw_avg += yaw;
                                    }
                                }
                                count_lock++;
                                if (count_lock == count_limit) {
                                    // collected enough
                                    human_avg /= float(count_lock);
                                    human_yaw_avg /= float(count_lock);
                                    while (human_yaw_avg <= -PI)
                                        human_yaw_avg += 2*PI;
                                    while (human_yaw_avg > PI)
                                        human_yaw_avg -= 2*PI;
                                    goal_human = human_avg + tf::Vector3(distance_to_goal*cos(human_yaw_avg), distance_to_goal*sin(human_yaw_avg), -0.4);
                                    q_goal.setRPY(0., 0., human_yaw_avg+PI);
                                    transform.setOrigin(goal_human);
                                    transform.setRotation(q_goal);
                                    goal_locked = true;         // lock the goal
                                    count_lock = 0;             // reset counter
                                }
                            }
                        }
                    }
                }
                catch (tf::TransformException &ex) {
                    ROS_ERROR("%s", ex.what());
                }
            }
        }
        
        void navigate(){
            if (!to_navigate)
                return;
            std::vector<std::string> frames;
            ros::Time last_time;
            std::string *error;
            try{
                listener.getFrameStrings(frames);
                if (std::find(frames.begin(), frames.end(), "blimp") != frames.end()){
                    listener.getLatestCommonTime("blimp","goal",last_time,error);
                    if (ros::Time::now()-last_time > ros::Duration(1))
                    {
                        prev_dist = 1e6;
                        prev_z = 1e6;
                        prev_angle = -2*PI;
                        return;         // Too old
                    }
                    //listener.lookupTransform("blimp", "goal", ros::Time(0), transform);
                    tf::StampedTransform t_blimp;
                    //listener.lookupTransform("blimp","goal_rotated", last_time, t_blimp);
                    listener.lookupTransform("blimp","goal", last_time, t_blimp);
                    tf::StampedTransform blimp_world;
                    listener.lookupTransform("world","blimp", ros::Time(0), blimp_world);
                    tf::Vector3 d_blimp = t_blimp.getOrigin();
                    
                    switch (state) {
                        case 0:         // Stop
                            // Check transition condition
                            
                            if (!inGoalVicinity(d_blimp)) {
                                if (fabs(d_blimp.y()) > 0.1) {     // If going direct --> > 10 cm lateral errorr --> need angle adjustment --> turn
                                    double y_turn = atan2(d_blimp.y(), d_blimp.x());
                                    //if (fabs(y_turn) > PI/2){
                                    //    command_yaw = yaw_blimp + y_turn - PI;        // Reversed
                                    //}else{
                                    //    command_yaw = yaw_blimp + y_turn;
                                    //}
                                    command_yaw = yaw_blimp + y_turn;
                                    if (command_yaw > PI)
                                        command_yaw -= 2*PI;
                                    else if (command_yaw <= -PI)
                                        command_yaw += 2*PI;

                                    // Set pinpoint
                                    pinpoint_tf.setOrigin(blimp_world.getOrigin());
                                    tf::Quaternion q_pin;
                                    q_pin.setRPY(0.,0.,command_yaw);
                                    pinpoint_tf.setRotation(q_pin);
                                    br.sendTransform(tf::StampedTransform(pinpoint_tf, ros::Time::now(), "world", "pinpoint"));
                                        
                                    std_msgs::Int16 cmd;
                                    cmd.data = round(-command_yaw*180./PI);
                                    cmd_yaw_pub.publish(cmd);
                                    //clearDrivePowers();
                                    state = 1;
                                    ROS_INFO("%d", state);
                                }
                                else {
                                    //clearDrivePowers();
                                    state = 2;
                                    ROS_INFO("%d", state);
                                }
                            }
                            else {
                                tf::Matrix3x3 m2(t_blimp.getRotation());
                                double r_turn, p_turn, y_turn;
                                m2.getRPY(r_turn, p_turn, y_turn);
                                if (fabs(y_turn) > PI/18.) {            // 10 degrees
                                    command_yaw = yaw_blimp + y_turn;
                                    if (command_yaw > PI)
                                        command_yaw -= 2*PI;
                                    else if (command_yaw <= -PI)
                                        command_yaw += 2*PI;

                                    // Set pinpoint at goal
                                    pinpoint_tf.setOrigin(transform.getOrigin());
                                    pinpoint_tf.setRotation(transform.getRotation());
                                    br.sendTransform(tf::StampedTransform(pinpoint_tf, ros::Time::now(), "world", "pinpoint"));
                                        
                                    ROS_INFO("command_yaw changed to %f", command_yaw);
                                    std_msgs::Int16 cmd;
                                    cmd.data = round(-command_yaw*180./PI);
                                    cmd_yaw_pub.publish(cmd);
                                    //clearDrivePowers();
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
                            if (fabs(yaw_blimp - command_yaw) <= PI/36. && stoppedRotating()) {     // 5 degrees
                                // Good alignment and already at low turning speed --> Stop
                                count_stop = 0;
                                state = 0;
                                ROS_INFO("%d", state);
                            }
                            else {
                                tf::StampedTransform b_pinpoint;
                                listener.lookupTransform("blimp","pinpoint", ros::Time(0), b_pinpoint);
                                tf::Vector3 error_blimp = b_pinpoint.getOrigin();
                                double r_turn, p_turn, y_turn;
                                // Need to adjust?
                                if (inGoalVicinity(error_blimp)) {
                                    tf::Matrix3x3 m2(b_pinpoint.getRotation());
                                    m2.getRPY(r_turn, p_turn, y_turn);
                                    // Also try to keep the x-y distance under control while rotating
                                    std_msgs::Float64 dist_msg;
                                    dist_msg.data = error_blimp.x();
                                    dist_pub.publish(dist_msg);
                                    std_msgs::Float64 dist_y_msg;
                                    dist_y_msg.data = error_blimp.y();
                                    dist_y_pub.publish(dist_y_msg);
                                }
                                else {
                                    y_turn = atan2(error_blimp.y(), error_blimp.x());
                                    clearDrivePowers();
                                }
                                float new_cmd = yaw_blimp + y_turn;
                                if (new_cmd > PI)
                                    new_cmd -= 2*PI;
                                else if (new_cmd <= -PI)
                                    new_cmd += 2*PI;
                                    
                                if (fabs(new_cmd - command_yaw) > PI/18.)  {    // difference of the command to the current situation (the blimp may drift and need to turn more/less to the goal)
                                    // Set new command and send
                                    command_yaw = new_cmd;
                                    ROS_INFO("command_yaw changed to %f", command_yaw);
                                    std_msgs::Int16 cmd;
                                    cmd.data = round(-command_yaw*180./PI);
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
            catch (tf::TransformException &ex){
                ROS_ERROR("%s", ex.what());
            }
        }
};


int main (int argc, char **argv) {
    ros::init(argc, argv, "blimp_controller");//, ros::init_options::AnonymousName);

    ros::NodeHandle nh_priv("~");
    std::string name;
    nh_priv.getParam("file_name", name);
    if (*(name.end() - 1) == 0x0a || *(name.end() - 1) == '\n')
        name.erase(name.end()-1);
    const std::string file_name = name;

    BlimpController blimp_controller(file_name);
    ros::Rate rate(10);
    while (ros::ok()){
        blimp_controller.publish();
        blimp_controller.navigate();
        ros::spinOnce();
        rate.sleep();
    }
    return 0;
}
