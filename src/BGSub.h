#ifndef BGSub_h
#define BGSub_h

#include "TrackedObject.h"
#include <opencv2/opencv.hpp>
#include "fern_based_classifier.h"

using namespace cv;

class BGSub {
public:
    BGSub(bool _toDraw, ofstream &_file, const char* outFileName, bool _toSave = false, bool _useFisheyeHOG = false);
    ~BGSub();
    bool processImage (Mat &input_img, bool detect_human=true);
protected:
    void groupContours ( vector< vector<Point> > inputContours, vector<RotatedRect> &outputBoundingBoxes, vector<RotatedRect> &rawBoundingBoxes, double distanceThreshold=1.0 );
    RotatedRect groupBlimp ( vector< vector<Point> > &inputContours, double distanceThreshold=1.0 );
    Size getHumanSize(float radius);
    void detectOriginalHOG(Mat &img, vector<RotatedRect> &ROIs, vector<RotatedRect> &detections, Size size_min, Size size_max, double scale0, int flag);
    void groupRectanglesNMS(vector<cv::RotatedRect>& rectList, vector<double>& weights, int groupThreshold, double overlapThreshold) const;

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
    
    // Fisheye HOG
    FisheyeHOGDescriptor hog_body;
    FisheyeHOGDescriptor hog_head;
    FisheyeHOGDescriptor hog_direction;

    // Original HOG
    HOGDescriptor hog_body_orig;
    HOGDescriptor hog_head_orig;
    HOGDescriptor hog_direction_orig;
    int imgBorder;

    HOGDescriptor hog_original;
    vector<TrackedObject> tracked_objects;
    vector<TrackedObject> tracked_humans;
    Point2f blimp_center;
    int hog_size;
    bool toDraw;
    
    fern_based_classifier * classifier;

    ofstream &f;
    long unsigned int count_img;
    bool useFisheyeHOG;

    float camHeight;
    float humanHeight;
    float humanWidth;
    float m;            // camera's m (scaling from metric to pixel
    float k1;           // first coefficient in fisheye model

    Mat circleMask;     // mask for marking the area that should be considered (eliminate flickering of codes in the corners)
};

#endif
