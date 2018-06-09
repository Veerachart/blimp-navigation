#include <ros/ros.h>
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/PolygonStamped.h>
#include "opencv2/opencv.hpp"
#include <tf/transform_broadcaster.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

using namespace geometry_msgs;
using namespace cv;

class Triangulator {
public:
    Triangulator(float _baseline=1.);
    void triangulateCallback(const PointStampedConstPtr& point_left, const PointStampedConstPtr& point_right);
    void triangulateHumanCallback(const PolygonStampedConstPtr& heads_left, const PolygonStampedConstPtr& heads_right);
    void update(void);
    
protected:
    ros::NodeHandle nh;
    tf::TransformBroadcaster br_;
    tf::Transform transform;
    tf::Quaternion q;
    tf::Transform transform_human;
    tf::Quaternion q_human;
    float baseline;
    
    typedef message_filters::sync_policies::ApproximateTime<PointStamped, PointStamped> MySyncPolicy;
    typedef message_filters::sync_policies::ApproximateTime<PolygonStamped, PolygonStamped> MySyncPolicy2;
    message_filters::Subscriber<geometry_msgs::PointStamped> sub_left, sub_right;
    message_filters::Subscriber<geometry_msgs::PolygonStamped> sub_human_left, sub_human_right;
    message_filters::Synchronizer<MySyncPolicy> sync;
    message_filters::Synchronizer<MySyncPolicy2> sync_human;
    
    KalmanFilter kf_blimp;
    bool is_tracking;
    
    KalmanFilter kf_human;
    bool is_tracking_human;
    int human_direction;
    
    float coeffs1[9], coeffs2[9];
    Mat R1, R2;
    float mu1, mv1, u01, v01, m1;
    float mu2, mv2, u02, v02, m2;
    float sigma_u;
    Mat sd_rotation_t;
    Mat sds;
    Mat sd_rotation;
    Mat sd_rotation_human_t;
    Mat sds_human;
    Mat sd_rotation_human;
    ros::Time last_time;
    ros::Time last_time_human;
};

Triangulator::Triangulator(float _baseline) : sub_left(nh, "cam_left/blimp_center", 1),
                                              sub_right(nh, "cam_right/blimp_center", 1),
                                              sub_human_left(nh, "cam_left/human_center", 1),
                                              sub_human_right(nh, "cam_right/human_center", 1),
                                              sync(Triangulator::MySyncPolicy(10), sub_left, sub_right), 
                                              sync_human(Triangulator::MySyncPolicy2(10), sub_human_left, sub_human_right) {
    //sub_left = message_filters::Subscriber<PointStamped>(nh, "cam_left/blimp_center", 1);
    //sub_right = message_filters::Subscriber<PointStamped>(nh, "cam_right/blimp_center", 1);
    //sync = message_filters::Synchronizer<MySyncPolicy> (MySyncPolicy(10), sub_left, sub_right);
    sync.registerCallback(boost::bind(&Triangulator::triangulateCallback, this, _1, _2));
    sync_human.registerCallback(boost::bind(&Triangulator::triangulateHumanCallback, this, _1, _2));
    
    baseline = _baseline;
    kf_blimp = KalmanFilter(6,3,0);
    is_tracking = false;
    q.setRPY(0,0,0);
    transform.setRotation(q);
    
    is_tracking_human = false;
    human_direction = -1;
    q_human.setRPY(0,0,0);
    transform_human.setRotation(q_human);
    
    setIdentity(kf_blimp.measurementMatrix);
    const Mat d = (Mat_<float>(1,6)<< 0.0009,0.0009,0.0009,0.0025,0.0025,0.0025);
    kf_blimp.processNoiseCov = Mat::diag(d);
    // Parameters of two cameras
    coeffs1[0] = -0.003125;
    coeffs1[2] =  0.001029;
    coeffs1[4] =  0.007671;
    coeffs1[6] =  0.013237;
    coeffs1[8] =  1.492357;
    coeffs1[1] = coeffs1[3] = coeffs1[5] = coeffs1[7] = 0.;
    coeffs2[0] = -0.003125;
    coeffs2[2] =  0.001029;
    coeffs2[4] =  0.007671;
    coeffs2[6] =  0.013237;
    coeffs2[8] =  1.492357;
    coeffs2[1] = coeffs2[3] = coeffs2[5] = coeffs2[7] = 0.;
    R1 = (Mat_<float>(3,3) <<  0.999601,  0.027689, -0.004448,
							  -0.027689,  0.999617,  0.000062,
							   0.004448,  0.000062,  0.999990);
    R2 = (Mat_<float>(3,3) <<  0.997059, -0.075636,  0.012355,
							   0.075567,  0.997123,  0.005980,
							  -0.012772, -0.005029,  0.999906);
							  
    mu1 = 157.1979;     
    mv1 = 157.2336;     
    u01 = 385.21;       
    v01 = 385.24;
    m1 = (mu1+mv1)/2.;
    mu2 = 156.4238;     
    mv2 = 156.4208;     
    u02 = 385.14;
    v02 = 385.32;
    m2 = (mu2+mv2)/2.;
    
    sigma_u = 5.;
    ////////////////////////////
}

void Triangulator::triangulateCallback(const PointStampedConstPtr& point_left, const PointStampedConstPtr& point_right) {
    ros::Time new_time;
    if (point_left->header.stamp - point_right->header.stamp > ros::Duration(0) )     // left after
        new_time = point_left->header.stamp;
    else
        new_time = point_right->header.stamp;
    
    if ( (point_left->point.x == 0 && point_left->point.y == 0) || (point_right->point.x == 0 && point_right->point.y == 0) ) {
        ROS_INFO("Not detected");
        return;
    }
    ROS_INFO("(%.2f, %.2f), (%.2f, %.2f)", point_left->point.x, point_left->point.y, point_right->point.x, point_right->point.y);
    
    float x,y,phi,r,theta;
    float psi1,beta1;
    float psi2,beta2;
    Mat u_cam, rect_r;
    
    // cam1
    x = (point_left->point.x-u01)/mu1;
    y = (point_left->point.y-v01)/mv1;
    phi = atan2(y,x);
    r = sqrt(x*x + y*y);
    theta = r/coeffs1[8];           // Linear r = k\theta
                                    // Or interpolation from the lookup table
    if (theta < 0 || theta > CV_PI/2) {
        ROS_INFO("Bad theta");
        return;
    }
    float c_theta1 = cos(theta), 
          s_theta1 = sin(theta),
          c_phi1   = cos(phi), 
          s_phi1   = sin(phi);
    
    u_cam = (Mat_<float>(3,1) << s_theta1*c_phi1, s_theta1*s_phi1, c_theta1);
    rect_r = R1*u_cam;
    psi1 = asin(rect_r.at<float>(0,0));
    beta1 = atan2(rect_r.at<float>(1,0), rect_r.at<float>(2,0));
    
    float J_theta1 = coeffs1[8];    // Linear
    Mat var_theta_phi1 = (Mat_<float>(2,2) << sigma_u*sigma_u/(m1*m1*J_theta1*J_theta1), 0.,
                                              0., sigma_u*sigma_u/(m1*m1*r*r));
    
    float c_psi1   = cos(psi1), 
          s_psi1   = sin(psi1), 
          c_beta1  = cos(beta1), 
          s_beta1  = sin(beta1);
          
    Mat J_ucam1 = (Mat_<float>(3,2) << c_theta1*c_phi1, -s_theta1*s_phi1,
                                       c_theta1*s_phi1,  s_theta1*c_phi1,
                                      -s_theta1,         0);
    
    Mat var_ucam1 = J_ucam1 * var_theta_phi1 * J_ucam1.t();
    
    Mat var_R1 = R1 * var_ucam1 * R1.t();
    
    Mat J_beta_psi1 = (Mat_<float>(2,3) << c_psi1, -s_psi1*s_beta1,        -s_psi1*c_beta1,
                                           0,       c_beta1/(c_psi1+1e-6), -s_beta1/(c_psi1+1e-6));
                                           
    Mat var_beta_psi1 = J_beta_psi1*var_R1*J_beta_psi1.t();
    
    // Cam2
    x = (point_right->point.x-u02)/mu2;
    y = (point_right->point.y-v02)/mv2;
    phi = atan2(y,x);
    r = sqrt(x*x + y*y);
    theta = r/coeffs2[8];           // Linear r = k\theta
                                    // Or interpolation from the lookup table
    if (theta < 0 || theta > CV_PI/2) {
        ROS_INFO("Bad theta");
        return;
    }
    
    float c_theta2 = cos(theta), 
          s_theta2 = sin(theta),
          c_phi2   = cos(phi), 
          s_phi2   = sin(phi);
    
    u_cam = (Mat_<float>(3,1) << s_theta2*c_phi2, s_theta2*s_phi2, c_theta2);
    rect_r = R2*u_cam;
    psi2 = asin(rect_r.at<float>(0,0));
    beta2 = atan2(rect_r.at<float>(1,0), rect_r.at<float>(2,0));
    
    float J_theta2 = coeffs2[8];    // Linear
    Mat var_theta_phi2 = (Mat_<float>(2,2) << sigma_u*sigma_u/(m2*m2*J_theta2*J_theta2), 0.,
                                              0., sigma_u*sigma_u/(m2*m2*r*r));
    
    float c_psi2   = cos(psi2), 
          s_psi2   = sin(psi2), 
          c_beta2  = cos(beta2), 
          s_beta2  = sin(beta2);
          
    Mat J_ucam2 = (Mat_<float>(3,2) << c_theta2*c_phi2, -s_theta2*s_phi2,
                                       c_theta2*s_phi2,  s_theta2*c_phi2,
                                      -s_theta2,         0);
    
    Mat var_ucam2 = J_ucam2*var_theta_phi2 * J_ucam2.t();
    
    Mat var_R2 = R2*var_ucam2*R2.t();
    
    Mat J_beta_psi2 = (Mat_<float>(2,3) << c_psi2, -s_psi2*s_beta2,        -s_psi2*c_beta2,
                                           0,       c_beta2/(c_psi2+1e-6), -s_beta2/(c_psi2+1e-6));
                                           
    Mat var_beta_psi2 = J_beta_psi2*var_R2*J_beta_psi2.t();
    
    ////////////////////////
    
    float disparity = psi1-psi2;
    
    if(fabs(beta1 - beta2) < 0.15) {        // On the same epipolar line
        if(disparity > 0) {
            float rho = baseline*c_psi2/sin(disparity);
            if (rho <= 10.) {
                float s_disp = sin(disparity), c_disp = cos(disparity);
                float x_out, y_out, z_out;
                x_out = rho*s_psi1;
                y_out = rho*c_psi1*s_beta1;
                z_out = rho*c_psi1*c_beta1;
                
                if (z_out >  0. && z_out < 3.5 &&
                    x_out > -1. && x_out < 5.  &&
                    y_out > -5. && y_out < 6.) {        // within the area
                    // calculate variance
                    Mat var_combi = Mat::zeros(4,4,CV_32FC1);
                    Mat aux = var_combi.colRange(0,2).rowRange(0,2);
                    var_beta_psi1.copyTo(aux);
                    aux = var_combi.colRange(2,4).rowRange(2,4);
                    var_beta_psi2.copyTo(aux);
                    
                    Mat J_p = Mat::zeros(3,4,CV_32FC1);
                    Mat temp = -baseline*c_psi2/s_disp *
                        (Mat_<float> (3,1) << -s_psi1*c_disp/s_disp + c_psi1,
                        -s_beta1*(c_psi1*c_disp/s_disp + s_psi1),
                        -c_beta1*(c_psi1*c_disp/s_disp + s_psi1));
                    temp.copyTo(J_p.col(0));
                    
                    temp = baseline*c_psi2/s_disp * 
                    (Mat_<float>(3,1) << 0, c_psi1*c_beta1, -c_psi1*s_beta1);
                    temp.copyTo(J_p.col(1));
                    
                    temp = baseline*(c_psi2*c_disp/(s_disp*s_disp) - s_psi2/s_disp) * 
                    (Mat_<float>(3,1) << s_psi1, c_psi1*s_beta1, c_psi1*c_beta1);
                    temp.copyTo(J_p.col(2));
                    
                    Mat var_p = J_p * var_combi * J_p.t();
                    
                    // use it to correct Kalman filter
                    if (is_tracking) {          // Already tracking something
                        // Check closeness to the tracked object
                        Mat measurement = (Mat_<float> (3,1) << x_out, y_out, z_out);
                        //std::cout << (measurement - kf_blimp.measurementMatrix*kf_blimp.statePost) << std::endl;
                        Mat x = sd_rotation*(measurement - kf_blimp.measurementMatrix*kf_blimp.statePost);
                        if (fabs(x.at<float>(0,0)) < 3*sds.at<float>(0,0) &&
                            fabs(x.at<float>(1,0)) < 3*sds.at<float>(1,0) &&
                            fabs(x.at<float>(2,0)) < 3*sds.at<float>(2,0)) {
                            // within 3 SD in all directions
                            kf_blimp.measurementNoiseCov = var_p;
                            kf_blimp.correct(measurement);
                            SVD::compute(kf_blimp.errorCovPost.colRange(0,3).rowRange(0,3), sds, sd_rotation, sd_rotation_t);
                        }
                        else {
                            ROS_INFO("Outside 3SD");
                        }
                    }
                    else {                      // Begin track;
                        kf_blimp.statePost = (Mat_<float>(6,1) << x_out, y_out, z_out, 0., 0., 0.);
                        setIdentity(kf_blimp.errorCovPost);
                        var_p.copyTo(kf_blimp.errorCovPost.colRange(0,3).rowRange(0,3));        // Set variance of x,y,z as computed, leave it to diag(1) for velocity
                        SVD::compute(kf_blimp.errorCovPost.colRange(0,3).rowRange(0,3), sds, sd_rotation, sd_rotation_t);
                        is_tracking = true;
                        ROS_INFO("Started tracking");
                    }
                    last_time = new_time;
                    return;
                }
            }
        }
    }
    ROS_INFO("No match");
    ROS_INFO("%.4f, %.4f, %.4f, %.4f", psi1, beta1, psi2, beta2);
}

void Triangulator::triangulateHumanCallback(const PolygonStampedConstPtr& heads_left, const PolygonStampedConstPtr& heads_right) {
    ros::Time new_time;
    if (heads_left->header.stamp - heads_right->header.stamp > ros::Duration(0) )     // left after
        new_time = heads_left->header.stamp;
    else
        new_time = heads_right->header.stamp;
    
    if ( !(heads_left->polygon.points.size()) || !(heads_right->polygon.points.size()) ) {
        ROS_INFO("Not detected");
        return;
    }
    
    float x,y,phi,r,theta;
    float psi1,beta1;
    float psi2,beta2;
    Mat u_cam, rect_r;
    
    // cam1
    x = (heads_left->polygon.points[0].x-u01)/mu1;
    y = (heads_left->polygon.points[0].y-v01)/mv1;
    phi = atan2(y,x);
    r = sqrt(x*x + y*y);
    theta = r/coeffs1[8];           // Linear r = k\theta
                                    // Or interpolation from the lookup table
    if (theta < 0 || theta > CV_PI/2) {
        ROS_INFO("Bad theta");
        return;
    }
    float c_theta1 = cos(theta), 
          s_theta1 = sin(theta),
          c_phi1   = cos(phi), 
          s_phi1   = sin(phi);
    
    u_cam = (Mat_<float>(3,1) << s_theta1*c_phi1, s_theta1*s_phi1, c_theta1);
    rect_r = R1*u_cam;
    psi1 = asin(rect_r.at<float>(0,0));
    beta1 = atan2(rect_r.at<float>(1,0), rect_r.at<float>(2,0));
    
    float J_theta1 = coeffs1[8];    // Linear
    Mat var_theta_phi1 = (Mat_<float>(2,2) << sigma_u*sigma_u/(m1*m1*J_theta1*J_theta1), 0.,
                                              0., sigma_u*sigma_u/(m1*m1*r*r));
    
    float c_psi1   = cos(psi1), 
          s_psi1   = sin(psi1), 
          c_beta1  = cos(beta1), 
          s_beta1  = sin(beta1);
          
    Mat J_ucam1 = (Mat_<float>(3,2) << c_theta1*c_phi1, -s_theta1*s_phi1,
                                       c_theta1*s_phi1,  s_theta1*c_phi1,
                                      -s_theta1,         0);
    
    Mat var_ucam1 = J_ucam1 * var_theta_phi1 * J_ucam1.t();
    
    Mat var_R1 = R1 * var_ucam1 * R1.t();
    
    Mat J_beta_psi1 = (Mat_<float>(2,3) << c_psi1, -s_psi1*s_beta1,        -s_psi1*c_beta1,
                                           0,       c_beta1/(c_psi1+1e-6), -s_beta1/(c_psi1+1e-6));
                                           
    Mat var_beta_psi1 = J_beta_psi1*var_R1*J_beta_psi1.t();
    
    // Cam2
    x = (heads_right->polygon.points[0].x-u02)/mu2;
    y = (heads_right->polygon.points[0].y-v02)/mv2;
    phi = atan2(y,x);
    r = sqrt(x*x + y*y);
    theta = r/coeffs2[8];           // Linear r = k\theta
                                    // Or interpolation from the lookup table
    if (theta < 0 || theta > CV_PI/2) {
        ROS_INFO("Bad theta");
        return;
    }
    
    float c_theta2 = cos(theta), 
          s_theta2 = sin(theta),
          c_phi2   = cos(phi), 
          s_phi2   = sin(phi);
    
    u_cam = (Mat_<float>(3,1) << s_theta2*c_phi2, s_theta2*s_phi2, c_theta2);
    rect_r = R2*u_cam;
    psi2 = asin(rect_r.at<float>(0,0));
    beta2 = atan2(rect_r.at<float>(1,0), rect_r.at<float>(2,0));
    
    float J_theta2 = coeffs2[8];    // Linear
    Mat var_theta_phi2 = (Mat_<float>(2,2) << sigma_u*sigma_u/(m2*m2*J_theta2*J_theta2), 0.,
                                              0., sigma_u*sigma_u/(m2*m2*r*r));
    
    float c_psi2   = cos(psi2), 
          s_psi2   = sin(psi2), 
          c_beta2  = cos(beta2), 
          s_beta2  = sin(beta2);
          
    Mat J_ucam2 = (Mat_<float>(3,2) << c_theta2*c_phi2, -s_theta2*s_phi2,
                                       c_theta2*s_phi2,  s_theta2*c_phi2,
                                      -s_theta2,         0);
    
    Mat var_ucam2 = J_ucam2*var_theta_phi2 * J_ucam2.t();
    
    Mat var_R2 = R2*var_ucam2*R2.t();
    
    Mat J_beta_psi2 = (Mat_<float>(2,3) << c_psi2, -s_psi2*s_beta2,        -s_psi2*c_beta2,
                                           0,       c_beta2/(c_psi2+1e-6), -s_beta2/(c_psi2+1e-6));
                                           
    Mat var_beta_psi2 = J_beta_psi2*var_R2*J_beta_psi2.t();
    
    ////////////////////////
    float disparity = psi1-psi2;
    
    if(fabs(beta1 - beta2) < 0.15) {        // On the same epipolar line
        if(disparity > 0) {
            float rho = baseline*c_psi2/sin(disparity);
            if (rho <= 10.) {
                float s_disp = sin(disparity), c_disp = cos(disparity);
                float x_out, y_out, z_out;
                x_out = rho*s_psi1;
                y_out = rho*c_psi1*s_beta1;
                z_out = rho*c_psi1*c_beta1;
                
                if (z_out >  0. && z_out < 3.5 &&
                    x_out > -1. && x_out < 5.  &&
                    y_out > -5. && y_out < 6.) {        // within the area
                    // calculate variance
                    Mat var_combi = Mat::zeros(4,4,CV_32FC1);
                    Mat aux = var_combi.colRange(0,2).rowRange(0,2);
                    var_beta_psi1.copyTo(aux);
                    aux = var_combi.colRange(2,4).rowRange(2,4);
                    var_beta_psi2.copyTo(aux);
                    
                    Mat J_p = Mat::zeros(3,4,CV_32FC1);
                    Mat temp = -baseline*c_psi2/s_disp *
                        (Mat_<float> (3,1) << -s_psi1*c_disp/s_disp + c_psi1,
                        -s_beta1*(c_psi1*c_disp/s_disp + s_psi1),
                        -c_beta1*(c_psi1*c_disp/s_disp + s_psi1));
                    temp.copyTo(J_p.col(0));
                    
                    temp = baseline*c_psi2/s_disp * 
                    (Mat_<float>(3,1) << 0, c_psi1*c_beta1, -c_psi1*s_beta1);
                    temp.copyTo(J_p.col(1));
                    
                    temp = baseline*(c_psi2*c_disp/(s_disp*s_disp) - s_psi2/s_disp) * 
                    (Mat_<float>(3,1) << s_psi1, c_psi1*s_beta1, c_psi1*c_beta1);
                    temp.copyTo(J_p.col(2));
                    
                    Mat var_p = J_p * var_combi * J_p.t();
                    
                    // use it to correct Kalman filter
                    if (is_tracking_human) {          // Already tracking something
                        // Check closeness to the tracked object
                        Mat measurement = (Mat_<float> (3,1) << x_out, y_out, z_out);
                        //std::cout << (measurement - kf_blimp.measurementMatrix*kf_blimp.statePost) << std::endl;
                        Mat x = sd_rotation_human*(measurement - kf_human.measurementMatrix*kf_human.statePost);
                        if (fabs(x.at<float>(0,0)) < 3*sds.at<float>(0,0) &&
                            fabs(x.at<float>(1,0)) < 3*sds.at<float>(1,0) &&
                            fabs(x.at<float>(2,0)) < 3*sds.at<float>(2,0)) {
                            // within 3 SD in all directions
                            kf_human.measurementNoiseCov = var_p;
                            kf_human.correct(measurement);
                            SVD::compute(kf_human.errorCovPost.colRange(0,3).rowRange(0,3), sds_human, sd_rotation_human, sd_rotation_human_t);
                        }
                        else {
                            ROS_INFO("Outside 3SD");
                        }
                    }
                    else {                      // Begin track;
                        kf_human.statePost = (Mat_<float>(6,1) << x_out, y_out, z_out, 0., 0., 0.);
                        setIdentity(kf_human.errorCovPost);
                        var_p.copyTo(kf_human.errorCovPost.colRange(0,3).rowRange(0,3));        // Set variance of x,y,z as computed, leave it to diag(1) for velocity
                        SVD::compute(kf_human.errorCovPost.colRange(0,3).rowRange(0,3), sds_human, sd_rotation_human, sd_rotation_human_t);
                        is_tracking_human = true;
                        ROS_INFO("Started tracking");
                    }
                    
                    // direction calculation
                    float direction_left = heads_left->polygon.points[0].z;
                    float direction_right = heads_right->polygon.points[0].z;
                    
                    float vx = kf_human.statePost.at<float>(3,0);
                    float vy = kf_human.statePost.at<float>(4,0);
                    
                    float prob_v_left = 1.f, prob_v_right = 1.f;
                    float prob_old_left = 1.f, prob_old_right = 1.f;
                    if (vx*vx + vy*vy > 0.01) {         // more than 10 cm/s
                        // get clue from direction of motion
                        float direction_v = atan2(vx, -vy) * 180./CV_PI;

                        float diff_left = abs(direction_left - direction_v);
                        
                        if (diff_left > 180)
                            diff_left = 360 - diff_left;
                        
                        if (diff_left <= 90)
                            prob_v_left = exp(-diff_left*diff_left/(2*30.*30.));
                        else
                            prob_v_left = 0;

                        float diff_right = abs(direction_right - direction_v);
                        
                        if (diff_right > 180)
                            diff_right = 360 - diff_right;
                        
                        if (diff_right <= 90)
                            prob_v_right = exp(-diff_right*diff_right/(2*30.*30.));           // P(velocity) * P(old direction)
                        else
                            prob_v_right = 0;
                    }
                    
                    if (human_direction >= 0) {         // Already tracking direction
                        float diff_old = abs(direction_left - human_direction);
                        if (diff_old > 180)
                            diff_old = 360 - diff_old;
                        prob_old_left = exp(-diff_old*diff_old/(2*15.*15.));
                        diff_old = abs(direction_right - human_direction);
                        if (diff_old > 180)
                            diff_old = 360 - diff_old;
                        prob_old_right = exp(-diff_old*diff_old/(2*15.*15.));
                    }
                    
                    float prob_left = prob_v_left * prob_old_left;
                    float prob_right = prob_v_right * prob_old_right;
                    
                    float d_l, d_r, direction_new;
                    if (prob_left + prob_right > 0) {
                        if (human_direction >= 0) {         // Merge with previous
                            if (direction_left - human_direction > 180)
                                d_l = direction_left - 360;
                            else if (human_direction - direction_left > 180)
                                d_l = direction_left + 360;
                            else
                                d_l = direction_left;
                            
                            if (direction_right - human_direction > 180)
                                d_r = direction_right - 360;
                            else if (human_direction - direction_right > 180)
                                d_r = direction_right + 360;
                            else
                                d_r = direction_right;
                            
                            direction_new = (prob_left*d_l + prob_right*d_r) / (prob_left + prob_right);
                            human_direction = (int) round((human_direction + direction_new)/2.);
                        }
                        else {                              // No tracking of direction yet
                            d_l = direction_left;
                            if (direction_right - direction_left > 180)
                                d_r = direction_right - 360;
                            else if (direction_left - direction_right > 180)
                                d_r = direction_right + 360;
                            else
                                d_r = direction_right;
                            
                            direction_new = (prob_left*d_l + prob_right*d_r) / (prob_left + prob_right);
                            human_direction = (int) round(direction_new);
                        }
                        while (human_direction < 0)
                            human_direction += 360;
                        while (human_direction >= 360)
                            human_direction += 360;
                        
                        q_human.setRPY(0,0,human_direction*CV_PI/180. - CV_PI/2.);
                    }
                    
                    last_time_human = new_time;
                    return;
                }
            }
        }
    }
}

void Triangulator::update(void) {
    ros::Time new_time = ros::Time::now();
    if (is_tracking) {
        double delta_t = (new_time - last_time).toSec();
        kf_blimp.transitionMatrix.at<float>(0,3) = delta_t;
        kf_blimp.transitionMatrix.at<float>(1,4) = delta_t;
        kf_blimp.transitionMatrix.at<float>(2,5) = delta_t;
        kf_blimp.predict();     // TODO use delta_t for update (velocity in [m/s])
        SVD::compute(kf_blimp.errorCovPost.colRange(0,3).rowRange(0,3), sds, sd_rotation, sd_rotation_t);
        transform.setOrigin(tf::Vector3(kf_blimp.statePost.at<float>(0,0),kf_blimp.statePost.at<float>(1,0),kf_blimp.statePost.at<float>(2,0)));
        transform.setRotation(q);
        br_.sendTransform(tf::StampedTransform(transform, new_time, "world", "blimp"));
    }
    if (is_tracking_human) {
        double delta_t = (new_time - last_time_human).toSec();
        kf_human.transitionMatrix.at<float>(0,3) = delta_t;
        kf_human.transitionMatrix.at<float>(1,4) = delta_t;
        kf_human.transitionMatrix.at<float>(2,5) = delta_t;
        kf_human.predict();     // TODO use delta_t for update (velocity in [m/s])
        SVD::compute(kf_human.errorCovPost.colRange(0,3).rowRange(0,3), sds, sd_rotation, sd_rotation_t);
        transform_human.setOrigin(tf::Vector3(kf_human.statePost.at<float>(0,0),kf_human.statePost.at<float>(1,0),kf_human.statePost.at<float>(2,0)));
        transform_human.setRotation(q_human);
        br_.sendTransform(tf::StampedTransform(transform_human, new_time, "world", "head"));
    }
    last_time = new_time;
    last_time_human = new_time;
}

int main (int argc, char **argv) {
    double baseline = 3.8;
    ros::init(argc, argv, "blimp_triangulator", ros::init_options::AnonymousName);
    ros::start();
    ros::Rate loop_rate(10);
    Triangulator triangulator(3.8);
    while(ros::ok()) {
        triangulator.update();
        ros::spinOnce();
        loop_rate.sleep();
    }
    
    return 0;
}
