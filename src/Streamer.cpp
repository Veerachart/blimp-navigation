#include <ros/ros.h>
#include <iostream>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <std_msgs/Bool.h>
#include "opencv2/opencv.hpp"

using namespace cv;

class Streamer {
    VideoCapture player1;
    VideoCapture player2;
    //ros::NodeHandle nh_;
    Mat frame1, frame2;
    //image_transport::ImageTransport it;
    image_transport::Publisher left_pub, right_pub;
    ros::Subscriber next_sub_l, next_sub_r;
    sensor_msgs::ImagePtr msg;
    bool leftFinished;
    bool rightFinished;
    
public:
    Streamer(){
        ros::NodeHandle nh_;
        ros::NodeHandle nh_priv_("~");
        image_transport::ImageTransport it(nh_);
        std::string left_vid_name, right_vid_name;
        if (!nh_priv_.hasParam("left_video") || !nh_priv_.hasParam("right_video") || !nh_priv_.hasParam("left_start_ms") || !nh_priv_.hasParam("right_start_ms")) {
            ROS_ERROR("USAGE: rosrun blimp_navigation stream_file_node _left_video:=left_video_file_name.mp4 _right_video:=right_video_file_name.mp4 _left_start_ms:=1000 _right_start_ms:=2000");
            ros::shutdown();
            return;
        }
        nh_priv_.getParam("left_video", left_vid_name);
        nh_priv_.getParam("right_video", right_vid_name);
        if(!player1.open(left_vid_name) || !player2.open(right_vid_name)){
            ROS_ERROR("Error opening video stream");
            ros::shutdown();
            return;
        }
        int left_start_ms, right_start_ms;
        nh_priv_.getParam("left_start_ms", left_start_ms);
        nh_priv_.getParam("right_start_ms", right_start_ms);
        player1.set(CV_CAP_PROP_POS_MSEC, left_start_ms);
        player2.set(CV_CAP_PROP_POS_MSEC, right_start_ms);
        left_pub = it.advertise("/cam_left/raw_video", 1);
        right_pub = it.advertise("/cam_right/raw_video", 1);
        next_sub_l = nh_.subscribe("/cam_left/next", 1, &Streamer::nextCallbackL, this);
        next_sub_r = nh_.subscribe("/cam_right/next", 1, &Streamer::nextCallbackR, this);
        leftFinished = false;
        rightFinished = false;
    }
    
    void nextCallbackL(const std_msgs::Bool::ConstPtr& flagMsg) {
        if (flagMsg->data) {
            leftFinished = true;
            if (rightFinished) {
                if(!player1.read(frame1) || !player2.read(frame2)){
                    ros::shutdown();
                    return;
                }
                msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", frame1).toImageMsg();
                left_pub.publish(msg);
                msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", frame2).toImageMsg();
                right_pub.publish(msg);

                leftFinished = false;
                rightFinished = false;
            }
        }
    }

    void nextCallbackR(const std_msgs::Bool::ConstPtr& flagMsg) {
        rightFinished = true;
        if (leftFinished) {
            if(!player1.read(frame1) || !player2.read(frame2)){
                ros::shutdown();
                return;
            }
            msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", frame1).toImageMsg();
            left_pub.publish(msg);
            msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", frame2).toImageMsg();
            right_pub.publish(msg);

            leftFinished = false;
            rightFinished = false;
        }
    }
};

int main(int argc, char **argv){
    ros::init(argc, argv, "file_streamer");
    Streamer streamer;
    ros::spin();
    return 0;
}
