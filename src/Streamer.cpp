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
    ros::NodeHandle nh_;
    Mat frame1, frame2;
    image_transport::ImageTransport it;
    image_transport::Publisher left_pub, right_pub;
    ros::Subscriber next_sub_l, next_sub_r;
    sensor_msgs::ImagePtr msg;
    
public:
    Streamer() : it(nh_) {
        if(!player1.open("/home/veerachart/Videos/CLIP_20170915-215436_left.mp4") || !player2.open("/home/veerachart/Videos/CLIP_20170915-215424_right.mp4")){
            std::cerr << "Error opening video stream" << std::endl;
            ros::shutdown();
            return;
        }
        player1.set(CV_CAP_PROP_POS_MSEC, 229050);
        player2.set(CV_CAP_PROP_POS_MSEC, 227000);
        left_pub = it.advertise("/cam_left/raw_video", 1);
        right_pub = it.advertise("/cam_right/raw_video", 1);
        next_sub_l = nh_.subscribe("/cam_left/next", 1, &Streamer::nextCallback, this);
        next_sub_r = nh_.subscribe("/cam_right/next", 1, &Streamer::nextCallback, this);
    }
    
    void nextCallback(const std_msgs::Bool::ConstPtr& flagMsg) {
        if (flagMsg->data) {
            if(!player1.read(frame1) || !player2.read(frame2)){
                ros::shutdown();
                return;
            }
            msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", frame1).toImageMsg();
            left_pub.publish(msg);
            msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", frame2).toImageMsg();
            right_pub.publish(msg);
        }
    }
};

int main(int argc, char **argv){
    ros::init(argc, argv, "file_streamer");
    Streamer streamer;
    ros::spin();
    return 0;
}
