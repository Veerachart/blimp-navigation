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
#include <std_msgs/Bool.h>

#include "TrackedObject.h"
#include "BGSub.h"

namespace enc = sensor_msgs::image_encodings;

class BlimpHumanTracker : public BGSub {
public:
    BlimpHumanTracker(std::ofstream &file, const char* file_name);
    void imageCallback(const sensor_msgs::Image::ConstPtr& msg);
    
private:
    image_transport::Subscriber image_sub_;
    image_transport::Publisher image_pub_;
    ros::Publisher center_pub_;
    ros::Publisher human_pub_;
    ros::Publisher next_pub_;
    std_msgs::Bool next_msg;
    
    geometry_msgs::PolygonStamped polygon;
    geometry_msgs::Polygon detected_points;
    geometry_msgs::Point32 point;
    geometry_msgs::PointStamped point_msg;
    
    bool detect_human;
};

BlimpHumanTracker::BlimpHumanTracker(std::ofstream &file, const char* file_name) :
        BGSub::BGSub(true, file, file_name, true, true) {
    ros::NodeHandle nh_;
    ros::NodeHandle nh_priv_("~");
    image_transport::ImageTransport it_(nh_);

    std::string camera (nh_.resolveName("camera"), 1, 5);
    //if (nh_.resolveName("save_video") == "/true")
    //    save_video = true;
    image_pub_ = it_.advertise("/cam_"+camera+"/detection_image", 1);
    image_sub_ = it_.subscribe("/cam_"+camera+"/raw_video", 1, &BlimpHumanTracker::imageCallback, this);
    
    //nh_.setParam("detect_human", false);
    //if (!nh_.hasParam("detect_human"))
    //    ROS_INFO("No parameter detect_human found");
    nh_priv_.param<bool>("detect_human", detect_human, true);
    
    center_pub_ = nh_.advertise<geometry_msgs::PointStamped>("/cam_"+camera+"/blimp_center", 1);
    human_pub_ = nh_.advertise<geometry_msgs::PolygonStamped>("/cam_"+camera+"/human_center", 1);
    next_pub_ = nh_.advertise<std_msgs::Bool>("/cam_"+camera+"/next",1);
    next_msg.data = true;
    ros::Duration(1.0).sleep();
    next_pub_.publish(next_msg);
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
    processImage(cv_ptr->image, detect_human);
    
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
    next_msg.data = true;
    next_pub_.publish(next_msg);
}


int main (int argc, char **argv) {
    std::cout << argv[0] << ", " << argv[1] << endl;
    ros::init(argc, argv, "blimp_human_tracker", ros::init_options::AnonymousName);
    //ros::NodeHandle nh();
    ros::NodeHandle nh_priv("~");
    string file_name;
    nh_priv.param<std::string>("file_name", file_name, "~/temp.csv");
    string csv_name = file_name + ".csv";
    std::ofstream file(csv_name.c_str());
    BlimpHumanTracker tracker = BlimpHumanTracker(file, file_name.c_str());
    ROS_INFO("START");
    ros::spin();
    
    return 0;
}
