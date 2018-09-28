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
#include <std_msgs/Bool.h>
#include <geometry_msgs/Pose.h>
#include <tf/transform_listener.h>
#include <fstream>

using namespace std;

const double PI = 3.141592653589793;

class VisualServo {
protected:
    ros::Publisher err_y_pub_;
    ros::Publisher err_size_pub_;
    ros::Publisher left_right_pub_;
    ros::Publisher yaw_pub_;
    ros::Publisher vis_serv_activator_pub_;         // For activating visual servo's PID controller
    ros::Publisher vis_serv_resetter_pub_;          // For resetting visual servo's PID controller
    ros::Publisher stereo_activator_pub_;           // For activating fisheye stereo's PID controller
    ros::Publisher stereo_resetter_pub_;            // For resetting fisheye stereo's PID controller
    ros::Publisher new_goal_from_blimp_pub_;        // For publishing the last face's direction before returning control to stereo
                                                    // So that the stereo will try to place the blimp in front of the face
    ros::Publisher reset_pub_;                      // Send out a msg when the face cannot be matched with the human for a while
    tf::TransformListener listener;

    std_msgs::Float64 msg_y, msg_size, msg_lr;
    std_msgs::Int16 msg_yaw;
    std_msgs::Bool false_msg;
    std_msgs::Bool true_msg;
    geometry_msgs::Pose new_goal_msg;

    ros::Subscriber face_sub_;
    ros::Subscriber yaw_sub_;

    ros::Time last_time;
    ros::Time last_humantf_time;

    tf::Vector3 last_human0;
    tf::Vector3 est_human;
    float new_goal_yaw;

    short current_yaw;

    int count_face_nomatch;

    bool isTrackingFace;
    void switchOnVisualServo();
    void switchOffVisualServo();
    bool isFaceLegit(int x, int y, int size);

    ofstream f_write;

public:
    VisualServo(string outputFileName);
    void faceCallback(const gpu_opencv::FaceServo::ConstPtr &msg);
    void yawCallback(const std_msgs::Int16::ConstPtr &msg);
    void run();
};

VisualServo::VisualServo(string outputFileName = "save.csv") : f_write(outputFileName.c_str(),std::ios::out) {
    ros::NodeHandle nh;
    face_sub_ = nh.subscribe("face", 1, &VisualServo::faceCallback, this);
    yaw_sub_ = nh.subscribe("yaw", 1, &VisualServo::yawCallback, this);

    err_y_pub_ = nh.advertise<std_msgs::Float64>("err_y", 1);
    err_size_pub_ = nh.advertise<std_msgs::Float64>("err_size", 1);
    left_right_pub_ = nh.advertise<std_msgs::Float64>("left_right", 1);
    yaw_pub_ = nh.advertise<std_msgs::Int16>("cmd_yaw", 1);

    vis_serv_activator_pub_ = nh.advertise<std_msgs::Bool>("vis_serv_enable", 1);
    vis_serv_resetter_pub_ = nh.advertise<std_msgs::Bool>("vis_serv_reset", 1);
    stereo_activator_pub_ = nh.advertise<std_msgs::Bool>("stereo_enable", 1);
    stereo_resetter_pub_ = nh.advertise<std_msgs::Bool>("stereo_reset", 1);
    new_goal_from_blimp_pub_ = nh.advertise<geometry_msgs::Pose>("new_goal_from_blimp", 1);

    reset_pub_ = nh.advertise<std_msgs::Bool>("face_reset",1);

    false_msg.data = false;
    true_msg.data = true;               // Preset the messages
    isTrackingFace = false;
    switchOffVisualServo();             // Starting with fisheye controller

    last_time = ros::Time::now();
    last_humantf_time = ros::Time(0);

    current_yaw = 0;
    count_face_nomatch = 0;
    f_write << "time,x,y,size,direction" << endl;
}

void VisualServo::faceCallback(const gpu_opencv::FaceServo::ConstPtr &msg) {
    last_time = msg->header.stamp;

    f_write << last_time << "," << msg->x << "," << msg->y << "," << msg->size << "," << msg->direction << endl;
    if (!isTrackingFace) {
        if (isFaceLegit(msg->x, msg->y, msg->size)) {
        //if (true) {
        //    isTrackingFace = true;
            ROS_INFO("The face is legit, start tracking this face.");
            count_face_nomatch = 0;
        }
        else {
            count_face_nomatch++;
            if (count_face_nomatch >= 10) {
                reset_pub_.publish(true_msg);
                count_face_nomatch = 0;
            }
            return;
        }
    }
    msg_y.data = (double) msg->y;
    err_y_pub_.publish(msg_y);
    msg_size.data = (double) 60./msg->size;
    err_size_pub_.publish(msg_size);

    int direction = msg->direction;         // 0: front, 1: left, -1:right

    // Get new goal
    if (listener.frameExists("blimp")) {
        ros::Time common_time;
        std::string *error;
        tf::StampedTransform transform;
        listener.getLatestCommonTime("world", "blimp", common_time, error);
        listener.lookupTransform("world", "blimp", common_time, transform);
        tf::Vector3 blimp = transform.getOrigin();
        double yaw = tf::getYaw(transform.getRotation());

        // offset of the face from the center --> angle offset
        double offset = msg->x * PI/1080.;
        // estimated human position from current blimp position and the face's position
        est_human = blimp + tf::Vector3(60./msg->size*cos(yaw+offset), 60./msg->size*sin(yaw+offset), 0.3*msg->y/msg->size + 0.6);      // 40 cm from balloon's center to camera (approx)
        // from the estimated human_position, set the goal

        tf::Vector3 new_goal = est_human + tf::Vector3(1.0*cos(yaw+offset+double(direction)*PI/4.+PI), 1.0*sin(yaw+offset+double(direction)*PI/4.+PI), -0.4);
        tf::Quaternion new_goal_q;
        new_goal_yaw = yaw+offset+direction*PI/4.;
        new_goal_q.setRPY(0, 0, new_goal_yaw);
        tf::pointTFToMsg(new_goal, new_goal_msg.position);
        tf::quaternionTFToMsg(new_goal_q, new_goal_msg.orientation);
    }

    msg_lr.data = (double) direction;
    left_right_pub_.publish(msg_lr);

    short yaw = -short(msg->x / 6.) + current_yaw;        // *60 degree / 360 px for half of the width
    while (yaw > 180)
        yaw -= 360;
    while(yaw <= -180)
        yaw += 360;
    if (abs(yaw - msg_yaw.data) > 5) {          // Changes more than 5 degree
        msg_yaw.data = yaw;
        yaw_pub_.publish(msg_yaw);
        ROS_INFO("Set yaw to %d", yaw);
    }
}

void VisualServo::yawCallback(const std_msgs::Int16::ConstPtr &msg) {
    current_yaw = msg->data;
}

void VisualServo::run() {
    tf::StampedTransform transform;
    try{
        if (listener.frameExists("human0")){
            ros::Time common_time;
            std::string *error;
            listener.getLatestCommonTime("world", "human0", common_time, error);
            if (common_time - last_humantf_time > ros::Duration(0.)) {
                listener.lookupTransform("world", "human0", common_time, transform);
                last_human0 = transform.getOrigin();
                last_humantf_time = common_time;
            }
        }
    }
    catch (tf::TransformException &ex){
        ROS_ERROR("%s", ex.what());
    }
    if (ros::Time::now() - last_time > ros::Duration(3.0)) {
        // not detected more than 2 seconds
        if (isTrackingFace) {
            // Turn off visual servo if it was running, else OK
            msg_y.data = 0.;
            err_y_pub_.publish(msg_y);
            msg_size.data = 60.;
            err_size_pub_.publish(msg_size);
            msg_lr.data = 0.;
            left_right_pub_.publish(msg_lr);
            msg_yaw.data = current_yaw;
            yaw_pub_.publish(msg_yaw);
            switchOffVisualServo();
            isTrackingFace = false;
        }
        //last_time = ros::Time::now();
    }
}

void VisualServo::switchOnVisualServo() {
    vis_serv_activator_pub_.publish(true_msg);
    stereo_resetter_pub_.publish(true_msg);
    stereo_activator_pub_.publish(false_msg);
    ROS_INFO("Switch to visual servo by the face!");
}

void VisualServo::switchOffVisualServo() {
    vis_serv_resetter_pub_.publish(true_msg);
    vis_serv_activator_pub_.publish(false_msg);
    stereo_activator_pub_.publish(true_msg);
    new_goal_from_blimp_pub_.publish(new_goal_msg);
    ROS_INFO("Estimated human at %.2f, %.2f, %.2f; New goal set at %.2f, %.2f, %.2f, %.2f", est_human.x(), est_human.y(), est_human.z(), new_goal_msg.position.x, new_goal_msg.position.y, new_goal_msg.position.z, new_goal_yaw*180./PI);
    ROS_INFO("Switched to fisheye stereo control!");
}

bool VisualServo::isFaceLegit(int x, int y, int size) {
    //isTrackingFace = true;
    //switchOnVisualServo();
    //return true;
    // Check if seeing a face at image's coordinate (x,y) is legit or not
    // By comparing the current position of the blimp and the human being tracked by fisheye stereo cameras
    if (last_humantf_time == ros::Time(0)) {    // No human found yet --> return
        isTrackingFace = false;
        return false;
    }

    std::vector<std::string> frames;
    tf::StampedTransform transform;
    try{
        listener.getFrameStrings(frames);
        if (std::find(frames.begin(), frames.end(), "blimp") != frames.end() ) { // &&
            //std::find(frames.begin(), frames.end(), "human0") != frames.end()){
            ros::Time common_time;
            std::string *error;
            //listener.getLatestCommonTime("blimp", "human0", common_time, error);
            //listener.lookupTransform("blimp", "human0", common_time, transform);
            listener.getLatestCommonTime("world", "blimp", common_time, error);
            listener.lookupTransform("world", "blimp", common_time, transform);


            tf::Matrix3x3 m(transform.getRotation());       // Rotation matrix

            tf::Vector3 blimp_to_human = m.transpose() * (last_human0 - transform.getOrigin());         // (x,y,z) of the last human body position in blimp's frame

            //cout << last_human0.x() << ", " << last_human0.y() << ", " << last_human0.z() << endl;
            //cout << blimp_to_human.x() << ", " << blimp_to_human.y() << ", " << blimp_to_human.z() << endl;

            float distance = sqrt(pow(blimp_to_human.x(),2) + pow(blimp_to_human.y(),2));               // Cartesian distance 2D

            if (blimp_to_human.x() < 0. || distance > 3.0) {
                // the person is behind, no way to see the face
                // the face is farther than 3.0 m, should not be large enough
                //ROS_INFO("Case 1: x = %.3f", distance);
                isTrackingFace = false;
                return false;
            }
            else {
                // need to check x,y
                // x corresponds with tf.y & y corresponds with tf.z
                //float est_face_size = max(-0.55*distance + 121., 24.);                  // Estimated face size based on the camera (min = 24)
                float est_face_size;
                if (distance >  1.0e-3) {
                    est_face_size = 60./distance;                               // Estimated face size based on pinhole camera model
                }
                else {
                    est_face_size = 150.;
                }

                if (fabsf(size - est_face_size) > 0.2 * est_face_size) {
                    // The tracked size is more than 20 % incorrect
                    //ROS_INFO("Case 1.5: est = %.3f, size = %d", est_face_size, size);
                    isTrackingFace = false;
                    return false;
                }

                if (blimp_to_human.z() > 1.6 && y < 0) {
                    // Blimp is higher, but the face is tracked in the upper half of the frame
                    //ROS_INFO("Case 2: z = %.3f, v = %d", transform.getOrigin().z(), y);
                    isTrackingFace = false;
                    return false;
                }

                double angle_face = atan(blimp_to_human.y()/blimp_to_human.x()) * 180./PI;
                if (fabs(angle_face) > 60.) {
                    // The person should not appear in the frame
                    //ROS_INFO("Case 3: angle = %.1f", angle_face);
                    isTrackingFace = false;
                    return false;
                }
                double yaw_img = double(x)/6.;

                if (fabs(angle_face - yaw_img) < 15.) {         // Location of the face and location from fisheye cameras are < 15 degree diff
                    //ROS_INFO("Case 4: angle = %.1f, yaw_img = %.1f", angle_face, yaw_img);
                    isTrackingFace = true;
                    switchOnVisualServo();
                    return true;
                }

                //ROS_INFO("Case 5: angle = %.1f, yaw_img = %.1f", angle_face, yaw_img);
                isTrackingFace = false;
                return false;
            }
        }
        else {
            //ROS_INFO("Case 6: tf not found");
            isTrackingFace = false;
            return false;
        }
    }
    catch (tf::TransformException &ex){
        ROS_ERROR("%s", ex.what());
    }

    isTrackingFace = false;
    return false;
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "visual_servo_node");
    ros::NodeHandle nh_private("~");
    string save_name;
    nh_private.getParam("filename", save_name);
    VisualServo servoNode(save_name);
    ros::Rate r(30);

    while (ros::ok()) {
        servoNode.run();
        ros::spinOnce();
        r.sleep();
    }
    //ros::spin();

    return 0;
}
