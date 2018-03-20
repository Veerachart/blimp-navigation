#ifndef BGSub_h
#define BGSub_h

#include "TrackedObject.h"
#include <opencv2/opencv.hpp>
#include "fern_based_classifier.h"

using namespace cv;

class BGSub {
public:
    BGSub(bool _toDraw, bool _toSave = false);
    ~BGSub();
    bool processImage (Mat &input_img);
protected:
    void groupContours ( vector< vector<Point> > inputContours, vector<RotatedRect> &outputBoundingBoxes, vector<RotatedRect> &rawBoundingBoxes, double distanceThreshold=1.0 );
    RotatedRect groupBlimp ( vector< vector<Point> > &inputContours, double distanceThreshold=1.0 );
    Size getHumanSize(float radius);

    Mat fgMaskMOG2;
    BackgroundSubtractorMOG2 pMOG2;
    
    Mat img_hsv;
    Mat temp;
    Mat img_thresholded_b;
    
    Mat show;
    Mat contour_show;
    float scale;
    
    //double u0, v0;
    
    double area_threshold;
    
    //std::ofstream *logfile;
    //double t_zero;
    
    //double f1,f2,f3;
    
    Point2f img_center;
    
    // Blue
    int iLowH_1;
    int iHighH_1;

    int iLowS_1;
    int iHighS_1;

    int iLowV_1;
    int iHighV_1;
    
    int dilation_size;
    
    VideoWriter outputVideo;
    bool save_video;
    
    FisheyeHOGDescriptor hog_body;
    FisheyeHOGDescriptor hog_head;
    FisheyeHOGDescriptor hog_direction;
    HOGDescriptor hog_original;
    vector<TrackedObject> tracked_objects;
    vector<TrackedObject> tracked_humans;
    int hog_size;
    bool toDraw;
    
    fern_based_classifier * classifier;
};

#endif
