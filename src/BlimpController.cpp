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
#include <geometry_msgs/Pose.h>
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
        ros::Subscriber yaw_sub;
        ros::Subscriber activate_sub;       // For subscribing the PID's activating topic to start/stop stereo control
        ros::Subscriber new_goal_from_blimp_sub;            // For new goal based on the blimp's camera; use for revising human's direction when the face is lost
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
        double goal_vicinity_approach;       // For checking when approaching to the goal (with FWD command)
        double goal_vicinity_maintain;       // For checking when trying to maintain position (with HOVER command).  maintain would be larger than approach
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
        bool goal_changed;          // used to mark that the goal has been changed (either manually or by human position change)
        float x_low, x_high;        // boundary of the area (x)
        float y_low, y_high;        // boundary of the area (y)
        bool using_stereo;          // Flag for switching on/off the controller based on fisheye stereo (changed by stereo_enable msg)
        ros::Time last_human_time;  // last time human0 is found in the tf
        
        std::ofstream file;          // file for saving goal position

        bool inGoalVicinity(tf::Vector3 distVec) {
            return sqrt(pow(distVec.x(),2) + pow(distVec.y(),2)) <= goal_vicinity_maintain;
        }

        bool arriveGoalVicinity(tf::Vector3 distVec) {
            return sqrt(pow(distVec.x(),2) + pow(distVec.y(),2)) <= goal_vicinity_approach;
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

        void limitGoalBoundary() {
            // Limit the goal within the experiment area (to avoid hitting the wall)
            // x_low        x_high
            //  ___________         y_low               -ve
            // |     2     |                            ----> 0
            // |1         3|                            |
            // |_____4_____|        y_high              |
            //     Door                                 V +PI/2
            // TODO
            if (outsideBoundary(goal_human)) {
                float delta_x, delta_y;
                float new_direction;
                bool finished = false;              // For returning -- cases which the person looks directly into the wall and cannot place the robot there
                if (goal_human.x() < x_low) {
                    // out side 1
                    if (human_yaw_avg > PI/2. && human_yaw_avg < 3.*PI/4.) {            // looking to front-left corner
                        delta_y = distance_to_goal;                                     // change to look to front
                        new_direction = -PI/2.;                                         // goal to the rear
                    }
                    else if (human_yaw_avg < -PI/2. && human_yaw_avg > -3.*PI/4.) {     // looking to rear-left corner
                        delta_y = -distance_to_goal;    // change to look to rear
                        new_direction = PI/2.;          // goal to the front
                    }
                    else {          // looking directly into the wall & therefore cannot see the face anyway
                        goal_human = transform.getOrigin();   // get the old one
                        q_goal = transform.getRotation();
                        finished = true;
                    }
                    delta_x = 0.;
                }
                else if (goal_human.x() > x_high) {
                    // out side 3
                    if (human_yaw_avg < -PI/4.) {       // looking to rear-right corner
                        delta_y = -distance_to_goal;    // change to look to rear
                        new_direction = PI/2.;          // goal to the front
                    }
                    else if (human_yaw_avg > PI/4.) {   // looking to front-right corner
                        delta_y = distance_to_goal;     // change to look to front
                        new_direction = -PI/2.;         // goal to the rear
                    }
                    else {                              // looking directly into the wall & therefore cannot see the face anyway
                        goal_human = transform.getOrigin();   // get the old one
                        q_goal = transform.getRotation();
                        finished = true;
                    }
                    delta_x = 0.;                       // No change for x
                }
                else if (goal_human.y() < y_low) {
                    // out side 2
                    if (human_yaw_avg < 0. && human_yaw_avg > -PI/4.) {             // looking to rear-right corner
                        delta_x = distance_to_goal;                                 // change to look to right
                        new_direction = -PI;                                        // goal to the left
                    }
                    else if (human_yaw_avg < -3.*PI/4. && human_yaw_avg > -PI) {    // looking to rear-left corner
                        delta_x = -distance_to_goal;                                // change to look to left
                        new_direction = 0.;                                         // goal to the right
                    }
                    else {                              // looking directly into the wall & therefore cannot see the face anyway
                        goal_human = transform.getOrigin();   // get the old one
                        q_goal = transform.getRotation();
                        finished = true;
                    }
                    delta_y = 0.;
                }
                else {
                    // out side 4
                    if (human_yaw_avg > 0. && human_yaw_avg < PI/4.) {              // looking to front-right corner
                        delta_x = distance_to_goal;                                 // change to look to right
                        new_direction = -PI;                                        // goal to the left
                    }
                    else if (human_yaw_avg > 3.*PI/4. && human_yaw_avg < PI) {      // looking to front-left corner
                        delta_x = -distance_to_goal;                                // change to look to left
                        new_direction = 0.;                                         // goal to the right
                    }
                    else {                              // looking directly into the wall & therefore cannot see the face anyway
                        goal_human = transform.getOrigin();   // get the old one
                        q_goal = transform.getRotation();
                        finished = true;
                    }
                    delta_y = 0.;
                }

                if (!finished) {
                    float new_x = human_avg.x() + delta_x;
                    float new_y = human_avg.y() + delta_y;
                    if (new_x > x_low && new_x < x_high && new_y > y_low && new_y < y_high) {       // Still within the boundary
                        goal_human.setX(new_x);
                        goal_human.setY(new_y);
                        q_goal.setRPY(0., 0., new_direction);
                        ROS_INFO("Adjusted goal looks good");
                    }
                    else {
                        // So what?
                        // TODO
                        ROS_INFO("New goal is outside the boundary: (%.4f, %.4f)", new_x, new_y);
                        if (new_x <= x_low) {
                            if (human_avg.x() - x_low > distance_to_goal - distance_border) {       // Still tolerable at the limit
                                goal_human.setX(x_low);
                                ROS_INFO("Distance reduced to %.2f", human_avg.x() - x_low);
                            }
                        }
                        else if (new_x >= x_high) {
                            if (x_high - human_avg.x() > distance_to_goal - distance_border) {      // Still tolerable at the limit
                                goal_human.setX(x_high);
                                ROS_INFO("Distance reduced to %.2f", x_high - human_avg.x());
                            }
                        }
                        else if (new_y <= y_low) {
                            if (human_avg.y() - y_low > distance_to_goal - distance_border) {       // Still tolerable at the limit
                                goal_human.setY(y_low);
                                ROS_INFO("Distance reduced to %.2f", human_avg.y() - y_low);
                            }
                        }
                        else if (new_y >= y_high) {
                            if (y_high - human_avg.y() > distance_to_goal - distance_border) {      // Still tolerable at the limit
                                goal_human.setY(y_high);
                                ROS_INFO("Distance reduced to %.2f", y_high - human_avg.y());
                            }
                        }
                        q_goal.setRPY(0., 0., new_direction);
                    }
                }
            }
            else {
                // Already good
                ROS_INFO("No problem!");
                q_goal.setRPY(0., 0., human_yaw_avg+PI);
            }
        }

        bool outsideBoundary(tf::Vector3 goal) {
            // Check if the goal is outside the boundary
            return (goal.x() < x_low) || (goal.x() > x_high) ||
                   (goal.y() < y_low) || (goal.y() > y_high);
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
            yaw_sub = nh_.subscribe("yaw", 1, &BlimpController::yawCallback, this);
            activate_sub = nh_.subscribe("stereo_enable", 1, &BlimpController::activatorCallback, this);
            new_goal_from_blimp_sub = nh_.subscribe("new_goal_from_blimp", 1, &BlimpController::newGoalFromBlimpCallback, this);
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
            goal_vicinity_maintain = 0.5;       // 50 cm
            goal_vicinity_approach = 0.3;       // 30 cm
            stop_criteria_rotate = 2.*PI/180.;
            stop_criteria_move = 0.005;      // 5 cm/s at 10 Hz
            distance_to_goal = 1.2;             // 1.5 m
            distance_border = 0.3;              // 30 cm
            angle_border = 0.2618;              // 15 degree
            count_lock = 0;
            count_limit = 10;                   // one second at 10 Hz
            count_stop = 0;
            goal_locked = false;
            to_navigate = false;
            goal_changed = false;
            using_stereo = true;                // Initiate to using the stereo pair for control
            human_yaw_avg = 0.f;
            x_low = -1.5;
            x_high = 4.7;
            y_low = -2.7;
            y_high = 2.7;

            file << "Time,state,goal_x,goal_y,goal_z,goal_yaw" << std::endl;
            file << std::fixed << std::setprecision(6);
        }
        
        void callback(blimp_navigation::ControllerConfig &config, uint32_t level){
            to_navigate = config.groups.goal.navigate;
            if (config.groups.goal.set) {
                goal = tf::Vector3(config.groups.goal.x, config.groups.goal.y, config.groups.goal.z);
                q.setRPY(0.0, 0.0, config.groups.goal.yaw*PI/180);
                transform.setOrigin(goal);
                transform.setRotation(q);
                goal_changed = true;
                br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "world", "goal"));
                ROS_INFO("Goal changed");
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
        
        void activatorCallback(const std_msgs::Bool::ConstPtr& msg) {
            using_stereo = msg->data;
            if (!using_stereo) {
                ROS_INFO("Switched off stereo.");
                prev_dist = 1e6;
                prev_z = 1e6;
                prev_angle = -2*PI;
                state = 0;
            }
        }

        void newGoalFromBlimpCallback(const geometry_msgs::Pose::ConstPtr& msg) {
            tf::Vector3 goal_origin;
            tf::Quaternion goal_quat;
            tf::pointMsgToTF(msg->position, goal_origin);
            tf::quaternionMsgToTF(msg->orientation, goal_quat);
            if (goal_origin.z() < 0.4)
                goal_origin.setZ(0.4);
            if (goal_origin.z() > 1.0)
                goal_origin.setZ(1.0);
            transform.setOrigin(goal_origin);
            transform.setRotation(goal_quat);
            goal_locked = true;             // We have set the goal, lock this
            ROS_INFO("Set goal to %.2f, %.2f, %.2f, with yaw %.2f", transform.getOrigin().x(), transform.getOrigin().y(), transform.getOrigin().z(), tf::getYaw(transform.getRotation())*180./PI);
        }

        void publish(){
            br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "world", "goal"));
            if (state == 1)
                br.sendTransform(tf::StampedTransform(pinpoint_tf, ros::Time::now(), "world", "pinpoint"));
            file << ros::Time::now().toSec() << "," << ((!to_navigate || !using_stereo) ? -1 : state) << "," << transform.getOrigin().x() << "," << transform.getOrigin().y() << "," << transform.getOrigin().z() << "," << tf::getYaw(transform.getRotation())*180./PI << std::endl;
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
                            //ROS_ERROR("Head too old");
                            return;
                        }
                        if (last_time == last_human_time) {
                            // The same time as previous loop --> ignore
                            return;
                        }
                        last_human_time = last_time;
                        listener.lookupTransform("world", "human0", last_time, human_tf);
                        tf::Vector3 human = human_tf.getOrigin();
                        tf::Quaternion q_human = human_tf.getRotation();
                        float yaw = tf::getYaw(q_human);
                        //ROS_INFO("Now: %.4f, %.4f, %.4f\t%.4f", human.x(), human.y(), human.z(), yaw*180./PI);
                        if (goal_locked) {
                            tf::StampedTransform check_tf;
                            ros::Time common_time;
                            listener.getLatestCommonTime("goal", "human0", common_time, error);
                            if (ros::Time::now() - common_time > ros::Duration(1.0)) {
                                ROS_ERROR("Head & goal not close enough in time");
                                if (goal_locked) {
                                    count_lock++;
                                }
                                if (count_lock == count_limit) {
                                    goal_locked = false;
                                    ROS_INFO("Goal unlocked");
                                }
                                return;
                            }
                            listener.lookupTransform("goal", "human0", common_time, check_tf);
                            tf::Vector3 check_vec = check_tf.getOrigin();
                            float yaw_diff = tf::getYaw(check_tf.getRotation());
                            while (yaw_diff > PI)
                                yaw_diff -= 2*PI;
                            while (yaw_diff <= -PI)
                                yaw_diff += 2*PI;
                            //std::cout << check_vec.x() << "," << check_vec.y() << "," << check_vec.z() << "\t" << yaw_diff*180./PI << std::endl;
                            if (check_vec.x() < distance_to_goal - distance_border ||
                                check_vec.x() > distance_to_goal + distance_border ||
                                check_vec.z() < 1.0 - distance_border ||
                                check_vec.z() > 1.0 + distance_border ||
                                fabs(atan2(check_vec.y(), check_vec.x())) > PI/4. ||
                                fabsf(yaw_diff) < 3.*PI/4. ) {
                                human_avg += human;
                                if (count_lock == 0) {
                                    // First one, no need to care about the sign
                                    human_yaw_avg += yaw;
                                }
                                else {
                                    // From the second, need to keep it on the same side as the first (wrapping problem)
                                    float current_avg = human_yaw_avg / float(count_lock);
                                    float diff = yaw - current_avg;
                                    float gap_plus, gap_minus;
                                    if (diff > PI) {
                                        gap_plus = PI - yaw;
                                        gap_minus = current_avg + PI;
                                        if (gap_plus >= gap_minus) {
                                            human_yaw_avg += yaw + 2*PI;
                                        }
                                        else {
                                            human_yaw_avg += yaw - 2*PI;
                                        }
                                    }
                                    else if (diff <= -PI) {
                                        gap_plus = PI - current_avg;
                                        gap_minus = yaw + PI;
                                        if (gap_plus >= gap_minus) {
                                            human_yaw_avg += yaw + 2*PI;
                                        }
                                        else {
                                            human_yaw_avg += yaw - 2*PI;
                                        }
                                    }
                                    else {
                                        human_yaw_avg += yaw;
                                    }
                                }
                                count_lock++;
                                //ROS_INFO("%d", count_lock);

                                if (count_lock == count_limit) {
                                    // reaverage goal
                                    human_avg /= float(count_lock);
                                    human_yaw_avg /= float(count_lock);
                                    count_lock = 0;
                                    while (human_yaw_avg <= -PI)
                                        human_yaw_avg += 2*PI;
                                    while (human_yaw_avg > PI)
                                        human_yaw_avg -= 2*PI;
                                    //goal_human = human_avg + tf::Vector3(distance_to_goal*cos(human_yaw_avg), distance_to_goal*sin(human_yaw_avg), -0.4);
                                    goal_human = human_avg + tf::Vector3(distance_to_goal*cos(human_yaw_avg), distance_to_goal*sin(human_yaw_avg), -1.0);       // From body's center
                                    // Limit the goal height not to go too high or too low
                                    if (goal_human.z() < 0.4)
                                        goal_human.setZ(0.4);
                                    if (goal_human.z() > 1.0)
                                        goal_human.setZ(1.0);
                                    // Boundary
                                    limitGoalBoundary();
                                    transform.setOrigin(goal_human);
                                    transform.setRotation(q_goal);
                                    human_avg = tf::Vector3(0.,0.,0.);  // reset
                                    human_yaw_avg = 0.f;        // reset
                                    count_lock = 0;             // reset counter
                                    goal_changed = true;
                                    ROS_INFO("Renew goal");
                                }
                            }
                            else {
                                // in good position, reset the counter and the averages
                                human_avg = tf::Vector3(0.,0.,0.);  // reset
                                human_yaw_avg = 0.f;        // reset
                                count_lock = 0;             // reset counter
                                //ROS_INFO("Still good");
                            }
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
                                    float current_avg = human_yaw_avg / float(count_lock);
                                    float diff = yaw - current_avg;
                                    float gap_plus, gap_minus;
                                    if (diff > PI) {
                                        gap_plus = PI - yaw;
                                        gap_minus = current_avg + PI;
                                        if (gap_plus >= gap_minus) {
                                            human_yaw_avg += yaw + 2*PI;
                                        }
                                        else {
                                            human_yaw_avg += yaw - 2*PI;
                                        }
                                    }
                                    else if (diff <= -PI) {
                                        gap_plus = PI - current_avg;
                                        gap_minus = yaw + PI;
                                        if (gap_plus >= gap_minus) {
                                            human_yaw_avg += yaw + 2*PI;
                                        }
                                        else {
                                            human_yaw_avg += yaw - 2*PI;
                                        }
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
                                    //goal_human = human_avg + tf::Vector3(distance_to_goal*cos(human_yaw_avg), distance_to_goal*sin(human_yaw_avg), -0.4);
                                    goal_human = human_avg + tf::Vector3(distance_to_goal*cos(human_yaw_avg), distance_to_goal*sin(human_yaw_avg), -1.0);       // From body's center
                                    // Limit the goal height not to go too high or too low
                                    if (goal_human.z() < 0.4)
                                        goal_human.setZ(0.4);
                                    if (goal_human.z() > 1.0)
                                        goal_human.setZ(1.0);
                                    // Boundary
                                    limitGoalBoundary();
                                    transform.setOrigin(goal_human);
                                    transform.setRotation(q_goal);
                                    human_avg = tf::Vector3(0.,0.,0.);  // reset
                                    human_yaw_avg = 0.f;        // reset
                                    goal_locked = true;         // lock the goal
                                    count_lock = 0;             // reset counter
                                    goal_changed = true;
                                    ROS_INFO("Goal set by human");
                                }
                            }
                        }
                    }
                    else {
                        // ROS_INFO("Human not detected");
                        if (goal_locked) {
                            count_lock++;
                        }
                        if (count_lock == count_limit) {
                            goal_locked = false;
                            // ROS_INFO("Goal unlocked");
                            return;
                        }
                    }
                }
                catch (tf::TransformException &ex) {
                    ROS_ERROR("%s", ex.what());
                }
            }
        }
        
        void navigate(){
            if (!to_navigate || !using_stereo)
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
                    listener.lookupTransform("blimp","goal", last_time, t_blimp);
                    tf::StampedTransform blimp_world;
                    listener.lookupTransform("world","blimp", ros::Time(0), blimp_world);
                    tf::Vector3 d_blimp = t_blimp.getOrigin();
                    
                    switch (state) {
                        case 0:         // Stop
                        {
                            // Check transition condition
                            
                            if (!inGoalVicinity(d_blimp)) {
                                if (fabsf(d_blimp.y()) > 0.2) {                // If going direct --> > 20 cm lateral error --> need angle adjustment --> turn
                                    double y_turn = atan2(d_blimp.y(), d_blimp.x());
                                    float new_cmd = yaw_blimp + y_turn;
                                    if (new_cmd > PI)
                                        new_cmd -= 2*PI;
                                    else if (new_cmd <= -PI)
                                        new_cmd += 2*PI;
                                    if (fabsf(new_cmd - command_yaw) > PI/18.) {         // difference of the command to the current situation (the blimp may drift and need to turn more/less to the goal)
                                        command_yaw = new_cmd;
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
                                        ROS_INFO("command_yaw changed to %f", command_yaw*180./PI);
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
                                        
                                    ROS_INFO("command_yaw changed to %f", command_yaw*180./PI);
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
                            if (goal_changed) {
                                goal_changed = false;
                                ROS_INFO("goal_changed reset");
                            }
                            
                            break;
                        }
                                
                        case 1:         // Turn
                        {
                            tf::StampedTransform b_pinpoint;
                            listener.lookupTransform("blimp","pinpoint", ros::Time(0), b_pinpoint);
                            tf::Vector3 error_blimp = b_pinpoint.getOrigin();
                            // Also try to keep the x-y distance under control while rotating
                            std_msgs::Float64 dist_msg;
                            dist_msg.data = error_blimp.x();
                            dist_pub.publish(dist_msg);
                            std_msgs::Float64 dist_y_msg;
                            dist_y_msg.data = error_blimp.y();
                            dist_y_pub.publish(dist_y_msg);
                            // Check transition condition
                            if (fabs(yaw_blimp - command_yaw) <= PI/36.) {     // 5 degrees
                                // Good alignment
                                if (stoppedRotating()) {
                                    // Already at low turning speed --> Stop
                                    count_stop = 0;
                                    state = 0;
                                    ROS_INFO("%d", state);
                                }
                            }
                            else {
                                count_stop = 0;
                                double r_turn, p_turn, y_turn;
                                if (goal_changed) {                         // Has someone changed the goal?
                                    if (inGoalVicinity(d_blimp)) {
                                        y_turn = command_yaw - yaw_blimp;
                                    }
                                    else {
                                        y_turn = atan2(d_blimp.y(), d_blimp.x());
                                    }
                                    goal_changed = false;
                                    ROS_INFO("goal_changed reset");
                                }
                                else if (inGoalVicinity(error_blimp)) {     // Need to adjust?
                                    tf::Matrix3x3 m2(b_pinpoint.getRotation());
                                    m2.getRPY(r_turn, p_turn, y_turn);
                                }
                                else {
                                    ROS_INFO("Out of the vicinity");
                                    if (inGoalVicinity(d_blimp)) {
                                        y_turn = command_yaw - yaw_blimp;
                                    }
                                    else {
                                        y_turn = atan2(d_blimp.y(), d_blimp.x());
                                    }
                                }

                                float new_cmd = yaw_blimp + y_turn;
                                if (new_cmd > PI)
                                    new_cmd -= 2*PI;
                                else if (new_cmd <= -PI)
                                    new_cmd += 2*PI;
                                if (fabsf(new_cmd - command_yaw) > PI/18. && fabsf(new_cmd - command_yaw) < 35.*PI/18.)  {    // difference of the command to the current situation (the blimp may drift and need to turn more/less to the goal)
                                    // Set new command and send
                                    command_yaw = new_cmd;
                                    ROS_INFO("command_yaw changed to %f", command_yaw*180./PI);

                                    // Set pinpoint
                                    pinpoint_tf.setOrigin(blimp_world.getOrigin());
                                    tf::Quaternion q_pin;
                                    q_pin.setRPY(0.,0.,command_yaw);
                                    pinpoint_tf.setRotation(q_pin);
                                    br.sendTransform(tf::StampedTransform(pinpoint_tf, ros::Time::now(), "world", "pinpoint"));
                                    //clearDrivePowers();

                                    std_msgs::Int16 cmd;
                                    cmd.data = round(-command_yaw*180./PI);
                                    cmd_yaw_pub.publish(cmd);
                                }
                                else {
                                    // Keep turning
                                }
                            }
                            
                            break;
                        }
                            
                        case 2:         // Move
                        {
                            // Check transition condition
                            if (arriveGoalVicinity(d_blimp)) {
                                // Close
                                // Still need to send commands to make it stop
                                std_msgs::Float64 dist_y_msg;
                                dist_y_msg.data = d_blimp.y();
                                dist_y_pub.publish(dist_y_msg);

                                // Moving algorithm
                                std_msgs::Float64 dist_msg;
                                dist_msg.data = d_blimp.x();
                                dist_pub.publish(dist_msg);

                                if (stoppedMoving(d_blimp)) {
                                    // Stop
                                    count_stop = 0;
                                    state = 0;
                                    ROS_INFO("%d", state);
                                }
                                // When close, should not attempt to adjust the angle anymore
                            }
                            else {
                                count_stop = 0;
                                if (fabs(d_blimp.y()) > 0.2) {     // If going direct --> > 20 cm lateral error --> need angle adjustment --> turn
                                    double y_turn = atan2(d_blimp.y(), d_blimp.x());
                                    float new_cmd = yaw_blimp + y_turn;
                                    if (new_cmd > PI)
                                        new_cmd -= 2*PI;
                                    else if (new_cmd <= -PI)
                                        new_cmd += 2*PI;

                                    if (fabsf(new_cmd - command_yaw) > PI/18. && fabsf(new_cmd - command_yaw) < 35.*PI/18.)  {    // difference of the command to the current situation (the blimp may drift and need to turn more/less to the goal)
                                        command_yaw = new_cmd;

                                        // Set pinpoint
                                        pinpoint_tf.setOrigin(blimp_world.getOrigin());
                                        tf::Quaternion q_pin;
                                        q_pin.setRPY(0.,0.,command_yaw);
                                        pinpoint_tf.setRotation(q_pin);
                                        br.sendTransform(tf::StampedTransform(pinpoint_tf, ros::Time::now(), "world", "pinpoint"));

                                        std_msgs::Int16 cmd;
                                        cmd.data = round(-command_yaw*180./PI);
                                        cmd_yaw_pub.publish(cmd);
                                        ROS_INFO("command_yaw changed to %f", command_yaw*180./PI);
                                        //clearDrivePowers();
                                        state = 1;
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
                            }
                            if (goal_changed) {
                                goal_changed = false;
                                ROS_INFO("goal_changed reset");
                            }
                            break;
                        }
                    }
                    // Altitude control
                    std_msgs::Float64 z_msg;
                    z_msg.data = d_blimp.z();
                    z_pub.publish(z_msg);
                    //ROS_INFO("z published");
                    
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
