#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
//#include <cvaux.h>
#include <math.h>
//#include <cxcore.h>
#include "opencv2/opencv.hpp"
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <geometry_msgs/Point32.h>
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/Polygon.h>
#include <geometry_msgs/PolygonStamped.h>
#include <algorithm>

#include "TrackedObject.h"
#include "BGSub.h"

namespace enc = sensor_msgs::image_encodings;

class BlimpHumanTracker : public BGSub {
public:
    BlimpHumanTracker();
    void imageCallback(const sensor_msgs::Image::ConstPtr& msg);
    
private:
    ros::NodeHandle nh_;
    image_transport::ImageTransport it_;
    image_transport::Subscriber image_sub_;
    image_transport::Publisher image_pub_;
    ros::Publisher center_pub_;
    ros::Publisher human_pub_;
    
    geometry_msgs::PolygonStamped polygon;
    geometry_msgs::Polygon detected_points;
    geometry_msgs::Point32 point;
    geometry_msgs::PointStamped point_msg;
};

BlimpHumanTracker::BlimpHumanTracker() : BGSub::BGSub(true, false), it_(nh_) {
    std::string camera (nh_.resolveName("camera"), 1, 5);
    //if (nh_.resolveName("save_video") == "/true")
    //    save_video = true;
    image_pub_ = it_.advertise("/cam_"+camera+"/detection_image", 1);
    image_sub_ = it_.subscribe("/cam_"+camera+"/raw_video", 1, &BlimpHumanTracker::imageCallback, this);
    
    center_pub_ = nh_.advertise<geometry_msgs::PointStamped>("/cam_"+camera+"/blimp_center", 1);
    human_pub_ = nh_.advertise<geometry_msgs::PolygonStamped>("/cam_"+camera+"/human_center", 1);
}

void BlimpHumanTracker::imageCallback (const sensor_msgs::Image::ConstPtr& msg) {
    // ROS Stuff
    detected_points = geometry_msgs::Polygon();
    cv_bridge::CvImagePtr cv_ptr;
    try
    {
        cv_ptr = cv_bridge::toCvCopy(msg, enc::BGR8);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }
    ///////////
    
    // Inherited processing
    processImage(cv_ptr->image);
    
    //ROS Stuff
    image_pub_.publish(cv_ptr->toImageMsg());
    for (int hum = 0; hum < tracked_humans.size(); hum++) {
        Point2f head = tracked_humans[hum].getPointHead();
        point.x = head.x;
        point.y = head.y;
        point.z = (float) tracked_humans[hum].getDirection();
        detected_points.points.push_back(point);
    }
    polygon.header.stamp = ros::Time::now();
    polygon.polygon = detected_points;
    human_pub_.publish(polygon);
    point_msg.header.stamp = ros::Time::now();
    if (blimp_center != Point2f(0,0)) {
        point_msg.point.x = blimp_center.x;
        point_msg.point.y = blimp_center.y;
        center_pub_.publish(point_msg);
    }
    waitKey(1);
}


int main (int argc, char **argv) {
    ros::init(argc, argv, "blimp_human_tracker", ros::init_options::AnonymousName);
    ros::start();
    BlimpHumanTracker tracker;
    ROS_INFO("START");
    ros::spin();
    
    return 0;
}
