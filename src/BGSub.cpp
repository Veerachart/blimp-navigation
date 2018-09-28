#include "BGSub.h"
#include "TrackedObject.h"
#include <opencv2/opencv.hpp>
#include <opencv/highgui.h>
#include <math.h>
#include <iostream>
#include "fern_based_classifier.h"
#include "BGSub.h"
#include <ros/package.h>

using namespace cv;
using namespace std;

bool compareContourAreas ( vector<Point> contour1, vector<Point> contour2 ) {
    double i = contourArea(Mat(contour1));
    double j = contourArea(Mat(contour2));
    return ( i > j );
}

BGSub::BGSub(bool _toDraw, ofstream &_file, const char* outFileName, bool _toSave, bool _useFisheyeHOG) : f(_file){
    area_threshold = 225;
    
    //logfile = new std::ofstream(buffer);
    //*logfile << "time,f1,f2,f3,ftot\n";
    
    // HSV color detect
    // Control window for adjusting threshold values
    
    iLowH_1 = 100;
    iHighH_1 = 110;

    iLowS_1 = 100;
    iHighS_1 = 255;

    iLowV_1 = 60;
    iHighV_1 = 240;
    
    namedWindow("Control", CV_WINDOW_AUTOSIZE); //create a window called "Control"
    
    cvCreateTrackbar("LowH1", "Control", &iLowH_1, 179); //Hue (0 - 179)
    cvCreateTrackbar("HighH1", "Control", &iHighH_1, 179);

    cvCreateTrackbar("LowS1", "Control", &iLowS_1, 255); //Saturation (0 - 255)
    cvCreateTrackbar("HighS1", "Control", &iHighS_1, 255);

    cvCreateTrackbar("LowV1", "Control", &iLowV_1, 255); //Value (0 - 255)
    cvCreateTrackbar("HighV1", "Control", &iHighV_1, 255);

    save_video = _toSave;
    if (save_video) {
        string videoName = string(outFileName);
        outputVideo.open(videoName + ".avi", CV_FOURCC('D','I','V','X'), 10, Size(768, 768), true);
        /*ColorVideo.open(videoName + "_blimp.avi", CV_FOURCC('D','I','V','X'), 10, Size(768, 768), true);
        BGVideo.open(videoName + "_BG.avi", CV_FOURCC('D','I','V','X'), 10, Size(768, 768), true);
        FGVideo.open(videoName + "_FG.avi", CV_FOURCC('D','I','V','X'), 10, Size(768, 768), true);
        FGHumanVideo.open(videoName + "_FGHuman.avi", CV_FOURCC('D','I','V','X'), 10, Size(768, 768), true);
        ObjVideo.open(videoName + "_Obj.avi", CV_FOURCC('D','I','V','X'), 10, Size(768, 768), true);
        DetectedVideo.open(videoName + "_Detected.avi", CV_FOURCC('D','I','V','X'), 10, Size(768, 768), true);*/
		if (!outputVideo.isOpened()) {
			cerr << "Could not write video." << endl;
			return;
		}
    }
    pMOG2 = BackgroundSubtractorMOG2(1000, 50.0, true);
    pMOG2.set("backgroundRatio", 0.75);
    pMOG2.set("fTau", 0.6);
    pMOG2.set("nmixtures", 3);
    pMOG2.set("varThresholdGen", 25.0);
    pMOG2.set("fVarInit", 36.0);
    pMOG2.set("fVarMax", 5*36.0);
    toDraw = _toDraw;
    useFisheyeHOG = _useFisheyeHOG;

    string path = ros::package::getPath("blimp_navigation");
    char classifier_name[128];
    strcpy(classifier_name, path.c_str());
    strcat(classifier_name, "/classifiers/classifier_acc_400-4");
	classifier = new fern_based_classifier(classifier_name);

	hog_size = classifier->hog_image_size;

    if (useFisheyeHOG) {
        //hog_body.setSVMDetector(FisheyeHOGDescriptor::getDefaultPeopleDetector());
        hog_body.load("/home/veerachart/HOG_Classifiers/32x64_weighted/cvHOGClassifier_32x64+hard.yaml");
        hog_head.load("/home/veerachart/HOG_Classifiers/head_fastHOG.yaml");
        hog_direction = FisheyeHOGDescriptor(Size(hog_size,hog_size), Size(hog_size/2,hog_size/2), Size(hog_size/4,hog_size/4), Size(hog_size/4,hog_size/4), 9);
    }

    else {
        hog_body_orig.load("/home/veerachart/HOG_Classifiers/32x64_weighted/cvHOGClassifier_32x64+hard.yaml");
        hog_head_orig.load("/home/veerachart/HOG_Classifiers/head_fastHOG.yaml");
        hog_direction_orig = HOGDescriptor(Size(hog_size,hog_size), Size(hog_size/2,hog_size/2), Size(hog_size/4,hog_size/4), Size(hog_size/4,hog_size/4), 9);
    }

    // **************************** //
    // REMARKS FOR RESULT RECORDING //
    // **************************** //
    // Frame#, TrackedObj#, TrackedHum#, DetectedObj#, DetectedHum#, DetectedHead#, // (cont.)
    //       , TrackedObjs, TrackedHums, DetectedObjs, DetectedHums, DetectedHeads, processTime\n
    //
    // Each TrackedObj/TrackedHum contains
    // countHuman, x, y, w, h, angle, x, y, w, h, angle, heightRatio, headRatio, deltaAngle
    //            |------ body -----||------ head -----| |----- head-body relationship ----|
    // Each Detected* contains
    // x, y, w, h, angle
    //
    // READING
    // Read line & the first 6 numbers to know TrackedObj#, TrackedHum#, DetectedObj#, DetectedHum#, DetectedHead#
    // Then use them to indicate how many data need to be read.
    count_img = 0;
    imgBorder = 14;

    camHeight = 2.6;
    humanHeight = 1.8;
    humanWidth = 0.5;
    m = 157.;
    k1 = 1.5;

    startMaskSet = false;
    startMaskInUse = false;
}

BGSub::~BGSub() {
	delete classifier;
}

bool BGSub::processImage (Mat &input_img, bool detect_human) {
    int64 start = getTickCount();
    if (img_center == Point2f() ) {
        img_center = Point2f(input_img.cols/2, input_img.rows/2);
        circleMask = Mat::zeros(input_img.size(),CV_8UC1);
        circle(circleMask, img_center, min(input_img.rows/2, input_img.cols/2)-imgBorder, Scalar(255), -1);
    }
    Mat original_img;
    input_img.copyTo(original_img);

    for (int track = 0; track < tracked_objects.size(); track++)
        tracked_objects[track].PredictObject();
    for (int track = 0; track < tracked_humans.size(); track++)
        tracked_humans[track].PredictObject();

    cvtColor(input_img, img_hsv, CV_BGR2HSV);
    //cvtColor(input_img, img_gray, CV_BGR2GRAY);

    //////////////// BLIMP DETECT PART ////////////////
    inRange(img_hsv, Scalar(iLowH_1, iLowS_1, iLowV_1), Scalar(iHighH_1, iHighS_1, iHighV_1), img_thresholded_b); //Threshold the image, Blue
    
    /*if (area_max > 0)
    {
        circle(cv_ptr->image, blimp_center, 6, Scalar(0,0,255));
        
        point_msg.header.stamp = ros::Time::now();
        point_msg.point.x = blimp_center.x;
        point_msg.point.y = blimp_center.y;
        center_pub_.publish(point_msg);
    }*/
    ////////////////////////////////////////////////


    pMOG2(input_img, fgMaskMOG2);
    fgMaskMOG2 &= circleMask;       // Mask the area outside the circle to remove flickering
    threshold(fgMaskMOG2, fgMaskMOG2, 128, 255, THRESH_BINARY);
    if (save_video) {
        Mat save = Mat::zeros(fgMaskMOG2.size(), CV_8UC3);
        cvtColor(fgMaskMOG2, save, CV_GRAY2BGR);
        //FGVideo << save;
    }
//    imshow("FG all", fgMaskMOG2);
    //imshow("Blimp Mask", img_thresholded_b);
    //outputVideo << save;
    
    //Mat intersect = img_thresholded_b & fgMaskMOG2;         // Blimp
    Mat intersect = img_thresholded_b;
    //fgMaskMOG2 -= intersect;                                // Human
    
    //morphologyEx(intersect, intersect, MORPH_OPEN, Mat::ones(3,3,CV_8U), Point(-1,-1), 1);
    morphologyEx(intersect, intersect, MORPH_CLOSE, Mat::ones(3,3,CV_8U), Point(-1,-1), 1);
    morphologyEx(intersect, intersect, MORPH_OPEN, Mat::ones(3,3,CV_8U), Point(-1,-1), 1);
    morphologyEx(intersect, intersect, MORPH_DILATE, Mat::ones(3,3,CV_8U), Point(-1,-1), 1);

    vector<vector<Point> > contours_blimp;
    findContours(intersect, contours_blimp, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
    
    RotatedRect blimp_bb;
    Mat mask_blimp = Mat::zeros(input_img.size(), CV_8UC1);
    vector< vector < Point > > hull(1);
    if (contours_blimp.size()) {
        std::sort(contours_blimp.begin(), contours_blimp.end(), compareContourAreas);
        blimp_bb = groupBlimp(contours_blimp, 0.25);
        blimp_center = blimp_bb.center;
        if (blimp_bb.center != Point2f(0.,0.) || blimp_bb.size != Size2f(0.,0.) || blimp_bb.angle != 0.) {
            convexHull(contours_blimp[0],hull[0]);
            drawContours(mask_blimp, hull, 0, Scalar(255), CV_FILLED);
        }
    }
    if (save_video) {
        Mat save = Mat::zeros(mask_blimp.size(), CV_8UC3);
        cvtColor(mask_blimp, save, CV_GRAY2BGR);
        Mat foreground;
        pMOG2.getBackgroundImage(foreground);
        //ColorVideo << save;
        //BGVideo << foreground;
    }
    morphologyEx(mask_blimp, mask_blimp, MORPH_DILATE, Mat::ones(9,9,CV_8U), Point(-1,-1), 1);      // Expand the blimp's mask to remove FG
    if (!startMaskSet) {
        mask_blimp.copyTo(blimpStartMask);
        startMaskSet = true;
        startMaskInUse = true;
        countBGMask = countNonZero(blimpStartMask);
    }
    else {
        if (startMaskInUse) {
            Mat BG, BGBlimp, BGBlimp_blue;
            pMOG2.getBackgroundImage(BG);
            BG.copyTo(BGBlimp, blimpStartMask);
            inRange(BGBlimp, Scalar(iLowH_1, iLowS_1, iLowV_1), Scalar(iHighH_1, iHighS_1, iHighV_1), BGBlimp_blue);
            int countBlueBG;
            countBlueBG = countNonZero(BGBlimp_blue);
            if (countBlueBG < 0.1*countBGMask) {
                // Has less than 10 percent of the BG model in the range of blimp's color --> we can stop removing FG by this mask
                startMaskInUse = false;
            }
        }
    }
//    imshow("Blimp", mask_blimp);
    fgMaskMOG2 = fgMaskMOG2 - mask_blimp;
    if (startMaskInUse)
        fgMaskMOG2 = fgMaskMOG2 - blimpStartMask;
                
    morphologyEx(fgMaskMOG2, fgMaskMOG2, MORPH_OPEN, Mat::ones(3,3,CV_8U), Point(-1,-1), 1);
    morphologyEx(fgMaskMOG2, fgMaskMOG2, MORPH_CLOSE, Mat::ones(5,5,CV_8U), Point(-1,-1), 2);
    morphologyEx(fgMaskMOG2, fgMaskMOG2, MORPH_OPEN, Mat::ones(3,3,CV_8U), Point(-1,-1), 1);
    morphologyEx(fgMaskMOG2, fgMaskMOG2, MORPH_DILATE, Mat::ones(3,3,CV_8U), Point(-1,-1), 1);
    if (save_video) {
        Mat save = Mat::zeros(fgMaskMOG2.size(), CV_8UC3);
        cvtColor(fgMaskMOG2, save, CV_GRAY2BGR);
        //FGHumanVideo << save;
    }
//    imshow("For human detect", fgMaskMOG2);
    
    //if (!detect_human)
    //    return false;
    vector<RotatedRect> humans;
    vector<RotatedRect> heads;
    vector<RotatedRect> objects, rawBoxes;
    vector<int> angles;
    vector<RotatedRect> area_heads;                 // ROI to search for heads = top half of objects
    if (detect_human) {

    vector<vector<Point> > contours_foreground;
    findContours(fgMaskMOG2.clone(), contours_foreground, CV_RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    //vector<RotatedRect> humans;
    //vector<RotatedRect> heads;
    vector<double> weights;
    vector<float> descriptors;

    //vector<RotatedRect> objects, rawBoxes;

    if(contours_foreground.size() > 0){
        std::sort(contours_foreground.begin(), contours_foreground.end(), compareContourAreas);
        
        double threshold = 1.2;
        groupContours(contours_foreground, objects, rawBoxes, threshold);


        if (objects.size()) {
        	Size size_min(1000,1000), size_max(0,0);
            Size size_head_min(1000,1000), size_head_max(0,0);
            vector<Size> sizes_min, sizes_max;
            vector<Size> heads_min, heads_max;
            for (int obj = 0; obj < objects.size(); obj++) {
                //cout << objects[obj].center << " " << objects[obj].size << " " << objects[obj].angle << "\t";
                Size temp = getHumanSize(norm(objects[obj].center - img_center));
                //Size temp_min = temp - Size(8,16);
                //Size temp_max = temp + Size(8,16);
                Size temp_min(cvRound(0.8*temp.width), 2*cvRound(0.8*temp.width));
                Size temp_max(cvRound(1.2*temp.width), 2*cvRound(1.2*temp.width));
                sizes_min.push_back(temp_min);
                sizes_max.push_back(temp_max);
                if (temp_min.width < size_min.width)
                    size_min = temp_min;
                if (temp_max.width > size_max.width)
                    size_max = temp_max;
                float head_width_min = max(6., 0.375*temp_min.width);
                float head_width_max = max(6., 0.6*temp_max.width);
                Size temp_head_min(head_width_min, head_width_min);
                Size temp_head_max(head_width_max, head_width_max);
                heads_min.push_back(temp_head_min);
                heads_max.push_back(temp_head_max);
                if (temp_head_min.width < size_head_min.width)
                    size_head_min = temp_head_min;
                if (temp_head_max.width > size_head_max.width)
                    size_head_max = temp_head_max;

        		float theta_r = objects[obj].angle*CV_PI/180.;
        		area_heads.push_back(RotatedRect(objects[obj].center + 0.25*objects[obj].size.height*Point2f(sin(theta_r), -cos(theta_r)), Size(objects[obj].size.width,objects[obj].size.height/2), objects[obj].angle));
        		//cout << objects[obj].center << " and " << area_heads.back().center << endl;
        	}

	        //cout << size_min << " " << size_max << " " << size_head_min << " " << size_head_max << endl;

            if (useFisheyeHOG)
                //hog_body.detectAreaMultiScale(input_img, objects, humans, weights, descriptors, size_min, size_max, 0., Size(4,2), Size(0,0), 1.05, 1.0);
                hog_body.detectAreaMultiScale2(input_img, objects, humans, weights, descriptors, sizes_min, sizes_max, size_min, size_max, 0., Size(4,2), Size(0,0), 1.05, 1.0);
            else {
                detectOriginalHOG(input_img, objects, humans, size_min, size_max, 1.05, 0);
            }

        	vector<bool> usedTrack(tracked_objects.size(),false), usedHumanTrack(tracked_humans.size(), false);
        	bool isHuman[objects.size()];
        	for (int obj = 0; obj < objects.size(); obj++)
        		isHuman[obj] = false;

        	vector<Point2f> intersect_points;
        	for (int hum = 0; hum < humans.size(); hum++) {
        		for (int obj = 0; obj < objects.size(); obj++) {
        			if ((rotatedRectangleIntersection(objects[obj], humans[hum], intersect_points)) != INTERSECT_NONE) {
        				isHuman[obj] = true;
        				break;
        			}
        		}

        		// Match detected human with tracked humans first
        		if (tracked_humans.size()) {
        			int best_track = 0;
			        float best_dist = 1000;
			        for (int track = 0; track < tracked_humans.size(); track++) {
				        if (usedHumanTrack[track])
					        continue;					// This track already got updated --> skip
				        float dist = tracked_humans[track].distToObject(humans[hum]);
				        if (dist < best_dist) {
					        best_track = track;
					        best_dist = dist;
				        }
			        }

			        if (best_dist < 3*tracked_humans[best_track].getSdBody()) {
				        // Update
				        //cout << "Update human with human:" << humans[hum].center << "," << humans[hum].size << " with " << tracked_humans[best_track].getBodyROI().center << "," << tracked_humans[best_track].getBodyROI().size << endl;
				        tracked_humans[best_track].UpdateObject(humans[hum], true);
				        usedHumanTrack[best_track] = true;
				        continue;
			        }
        		}

		        // Then with tracked objects
		        if (tracked_objects.size()) {
			        int best_track = 0;
			        float best_dist = 1000;
			        int best_count = 0;			// Keep the value of countHuman of the best object -- give priority to the object with more count of human
			        for (int track = 0; track < tracked_objects.size(); track++) {
				        if (usedTrack[track])
					        continue;					// This track already got updated --> skip
				        float dist = tracked_objects[track].distToObject(humans[hum]);
				        int count = tracked_objects[track].getCount();
				        if (dist < best_dist && count >= best_count) {
					        best_track = track;
					        best_dist = dist;
				        }
			        }

			        if (best_dist < 3*tracked_objects[best_track].getSdBody()) {
				        // Update
				        //cout << "Update object with human:" << humans[hum].center << "," << humans[hum].size << " with " << tracked_objects[best_track].getBodyROI().center << "," << tracked_objects[best_track].getBodyROI().size << endl;
				        tracked_objects[best_track].UpdateObject(humans[hum], true);
				        if(tracked_objects[best_track].getStatus() == HUMAN) {
					        // Converted to human after update
					        // Take out from tracked_objects & add to tracked_human
					        tracked_humans.push_back(tracked_objects[best_track]);
					        tracked_objects.erase(tracked_objects.begin() + best_track);
					        usedHumanTrack.push_back(true);		// Needed?
					        usedTrack.erase(usedTrack.begin() + best_track);
					        //cout << "Upgrade object to human" << endl;
					        continue;
				        }
				        else
					        usedTrack[best_track] = true;
			        }
			        else {
				        // Not within range for the existing object, create a new one
				        //cout << "Added new object, starting as human." << endl;
				        tracked_objects.push_back(TrackedObject(humans[hum], true, false, RotatedRect(), img_center));
				        usedTrack.push_back(true);				// Needed?
			        }
		        }
		        else {
			        //cout << "Added new object, starting as human." << endl;
			        tracked_objects.push_back(TrackedObject(humans[hum], true, false, RotatedRect(), img_center));
			        usedTrack.push_back(true);				// Needed?
		        }
        	}

        	for (int obj = 0; obj < objects.size(); obj++) {
        		if (!isHuman[obj]) {
        			// This object is not marked as a human yet, so check it as an object
        			// The same way, match with human first
        			if (tracked_humans.size()) {
				        int best_track = 0;
				        float best_dist = 1000;
				        for (int track = 0; track < tracked_humans.size(); track++) {
					        if (usedHumanTrack[track])
						        continue;					// This track already got updated --> skip
					        float dist = tracked_humans[track].distToObject(objects[obj]);
					        if (dist < best_dist) {
						        best_track = track;
						        best_dist = dist;
					        }
				        }

				        if (best_dist < 3*tracked_humans[best_track].getSdBody()) {
					        // Update
					        //cout << "Update human with object:" << objects[obj].center << "," << objects[obj].size << " with " << tracked_humans[best_track].getBodyROI().center << "," << tracked_humans[best_track].getBodyROI().size << endl;
					        tracked_humans[best_track].UpdateObject(objects[obj], false);
					        usedHumanTrack[best_track] = true;
					        continue;
				        }
			        }

        			if (tracked_objects.size()) {
				        int best_track = 0;
				        float best_dist = 1000;
				        int best_count = 0;
				        for (int track = 0; track < tracked_objects.size(); track++) {
					        if (usedTrack[track])
						        continue;					// This track already got updated --> skip
					        float dist = tracked_objects[track].distToObject(objects[obj]);
					        int count = tracked_objects[track].getCount();
					        if (dist < best_dist && count >= best_count) {
						        best_track = track;
						        best_dist = dist;
					        }
				        }

				        if (best_dist < 3*tracked_objects[best_track].getSdBody()) {
					        // Update
					        //cout << "Update object with object:" << objects[obj].center << "," << objects[obj].size << " with " << tracked_objects[best_track].getBodyROI().center << "," << tracked_objects[best_track].getBodyROI().size << endl;
					        tracked_objects[best_track].UpdateObject(objects[obj], false);
					        usedTrack[best_track] = true;
				        }
				        else {
					        // Not within range for the existing object, create a new one
					        //cout << "Added new object, not containing human." << endl;
					        tracked_objects.push_back(TrackedObject(objects[obj], false, false, RotatedRect(), img_center));
					        usedTrack.push_back(true);		// Needed?
				        }
			        }
        			else {
				        // New object
				        //cout << "Added new object, not containing human." << endl;
				        tracked_objects.push_back(TrackedObject(objects[obj], false, false, RotatedRect(), img_center));
				        usedTrack.push_back(true);			// Needed?
			        }
        		}
        	}

        	/*input_img.copyTo(draw);
        	for (int track = 0; track < tracked_objects.size(); track++) {
		        Scalar color(255,255,255);
		        TrackedObject object = tracked_objects[track];
		        if (object.getStatus() == HUMAN)
			        color = Scalar(0,255,0);
		        circle(draw, object.getPointBody(), 3, color, -1);
		        Point2f rect_points[4];
		        tracked_objects[track].getBodyROI().points(rect_points);
		        for(int j = 0; j < 4; j++)
			        line( draw, rect_points[j], rect_points[(j+1)%4], Scalar(192,192,0),2,8);
		        //circle(input_img, object.getPointBody(), 3*object.getSdBody(), Scalar(192,192,0), 2);

		        //circle(input_img, object.getPointHead(), 2, Scalar(192,192,192), -1);
		        tracked_objects[track].getHeadROI().points(rect_points);
		        for(int j = 0; j < 4; j++)
			        line( draw, rect_points[j], rect_points[(j+1)%4], Scalar(0,192,192),2,8);
		        circle(draw, object.getPointHead(), 3*object.getSdHead(), Scalar(0,192,192), 2);
	        }
	        imshow("After human detect", draw);*/

            for (int track = 0; track < tracked_humans.size(); track++) {
                tracked_humans[track].updateHeadfromBody();
            }
            for (int track = 0; track < tracked_objects.size(); track++) {
                tracked_objects[track].updateHeadfromBody();
            }

            if (useFisheyeHOG)
                //hog_head.detectAreaMultiScale(input_img, area_heads, heads, weights, descriptors, size_head_min, size_head_max, 8.3, Size(4,2), Size(0,0), 1.05, 1.0);
                hog_head.detectAreaMultiScale2(input_img, area_heads, heads, weights, descriptors, heads_min, heads_max, size_head_min, size_head_max, 8.3, Size(4,2), Size(0,0), 1.05, 1.0);
            else
                detectOriginalHOG(input_img, area_heads, heads, size_min, size_max, 1.05, 1);


        	vector<bool> humanHasHead(tracked_humans.size(), false);
        	vector<bool> objectHasHead(tracked_objects.size(), false);
        	for (int head = 0; head < heads.size(); head++) {
        		if(tracked_humans.size()) {
			        int best_track = 0;
			        float best_dist = 1000;
			        for (int track = 0; track < tracked_humans.size(); track++) {
				        if (humanHasHead[track])
					        continue;					// already got head updated
				        float dist = tracked_humans[track].distToHead(heads[head]);
				        if (dist < best_dist) {
					        best_track = track;
					        best_dist = dist;
				        }
			        }
			        if (best_dist < 3*tracked_humans[best_track].getSdHead() || tracked_humans[best_track].isTrackedHeadInvalid()) {
				        // Update
				        //cout << "Update head:" << heads[head].center << "," << heads[head].size << " with " << tracked_humans[best_track].getBodyROI().center << "," << tracked_humans[best_track].getBodyROI().size << endl;
			            if (tracked_humans[best_track].isNewHeadLegit(heads[head])){
                            tracked_humans[best_track].UpdateHead(heads[head]);
                            humanHasHead[best_track] = true;
                            continue;
			            }
			        }
			        else {
				        // Outside 3 SD, should we add a new object here?
				        // TODO
			        }
        		}
        		if(tracked_objects.size()) {
        			int best_track = 0;
			        float best_dist = 1000;
			        for (int track = 0; track < tracked_objects.size(); track++) {
				        if (objectHasHead[track])
					        continue;					// already got head updated
				        float dist = tracked_objects[track].distToHead(heads[head]);
				        if (dist < best_dist) {
					        best_track = track;
					        best_dist = dist;
				        }
			        }

			        if (best_dist < 3*tracked_objects[best_track].getSdHead() || tracked_objects[best_track].isTrackedHeadInvalid()) {
				        // Update
				        //cout << "Update head:" << heads[head].center << "," << heads[head].size << " with " << tracked_objects[best_track].getBodyROI().center << "," << tracked_objects[best_track].getBodyROI().size << endl;
                        if (tracked_objects[best_track].isNewHeadLegit(heads[head])){
                            tracked_objects[best_track].UpdateHead(heads[head]);                // <----- OCCURS HERE!
                            objectHasHead[best_track] = true;
                            continue;
                        }
			        }
			        else {
				        // Outside 3 SD, should we add a new object here?
				        // TODO
			        }
		        }
        	}

        	for (int track = 0; track < tracked_humans.size(); track++) {
		        if (!humanHasHead[track]) {
			        // If this track has not got any update for head
			        // What should we do? TODO
			        if (tracked_humans[track].getStatus() == HUMAN) {
				        // If we are tracking this as HUMAN, head should be approximated
				        // TODO
			        }
		        }
	        }

        	for (int track = 0; track < tracked_objects.size(); track++) {
        		if (!objectHasHead[track]) {
        			// If this track has not got any update for head
        			// What should we do? TODO
        			if (tracked_objects[track].getStatus() == HUMAN) {
        				// If we are tracking this as HUMAN, head should be approximated
        				// TODO
        			}
        		}
        	}
        }
    }

    vector<float> descriptor;
    int output_class, output_angle;
    vector<int> classes;//, angles;
    //cout << "Current objects:" << endl;
    // Final clean up for large variance objects
    for (vector<TrackedObject>::iterator it = tracked_objects.begin(); it != tracked_objects.end(); ) {
        //cout << "\t" << it->getBodyROI().center << "," << it->getBodyROI().size << endl;
        if (it->CheckAndDelete()) {
	        it = tracked_objects.erase(it);
	        cout << "An object removed" << endl;
        }
        else
	        ++it;
    }
    //cout << "Current humans:" << endl;
    for (vector<TrackedObject>::iterator it = tracked_humans.begin(); it != tracked_humans.end(); ) {
        //cout << "\t" << it->getBodyROI().center << "," << it->getBodyROI().size << endl;
        if (it->CheckAndDelete()) {
	        it = tracked_humans.erase(it);
	        cout << "A human removed" << endl;
        }
        else {
	        // Calculate Ferns & direction
	        if (it->getStatus() == HUMAN) {
		        //Point2f head_vel = it->getHeadVel();
	            Point2f head_vel = it->getBodyVel();
		        RotatedRect rect = it->getHeadROI();

		        float walking_dir;
		        if (norm(head_vel) < it->getBodyROI().size.width/20.) {				// TODO Threshold adjust
			        walking_dir = -1.;					// Not enough speed, no clue
		        }
		        else {
                    //walking_dir = rect.angle + atan2(head_vel.x, head_vel.y)*180./CV_PI;          // Estimated walking direction relative to the radial line (0 degree head direction)
                    walking_dir = 180. + rect.angle - atan2(head_vel.x, -head_vel.y)*180./CV_PI;
                    while (walking_dir < 0.)
                        walking_dir += 360.;                    // [0, 360) range. Negative means no clue
                    while (walking_dir >= 360.)
                        walking_dir -= 360.;
		        }
		        //cout << head_vel << " Moving in " << walking_dir << " degree direction" << endl;
		        //cout << rect.center << " " << rect.size << " " << rect.angle << endl;

		        // Check vertices within frame
		        Point2f vertices[4];
		        rect.points(vertices);
		        int v;
		        for (v = 0; v < 4; v++) {
			        if (vertices[v].x < 0 || vertices[v].x >= input_img.cols || vertices[v].y < 0 || vertices[v].y >= input_img.rows) {
				        break;
			        }
		        }
		        if (v < 4) {					// At least one out
			        //cout << "OB" << endl;
			        classes.push_back(-1);
			        angles.push_back(-1);
			        it->updateDirection(-1, int(cvRound(walking_dir)));
			        ++it;
			        continue;
		        }
		        // crop head area
		        Mat M = getRotationMatrix2D(rect.center, rect.angle, 1.0);
		        Mat rotated, cropped;
		        warpAffine(original_img, rotated, M, original_img.size(), INTER_CUBIC);
		        getRectSubPix(rotated, rect.size, rect.center, cropped);
		        resize(cropped, cropped, Size(hog_size,hog_size));
		        /////////////////
                if (useFisheyeHOG) {
                    vector<RotatedRect> location;
                    location.push_back(rect);
                    hog_direction.compute(original_img, descriptor, location);
                }
                else {
                    hog_direction_orig.compute(cropped, descriptor);
                }
		        classifier->recognize_interpolate(descriptor, cropped, output_class, output_angle, walking_dir);
		        it->updateDirection(output_angle, int(cvRound(walking_dir)));
		        classes.push_back(output_class);
		        angles.push_back(output_angle);
	        }
	        else{
		        classes.push_back(-1);
		        angles.push_back(-1);
	        }
	        ++it;
        }
    }
    }
    int64 total_time = getTickCount() - start;

        /*for(int i = 0; i < contours_foreground.size(); i++){
            double area = contourArea(contours_foreground[i]);
            if(area < area_threshold){        // Too small contour
                continue;
            }
            RotatedRect rect;
            rect = minAreaRect(contours_foreground[i]);
            Point2f rect_points[4];
            rect.points(rect_points);
            for(int j = 0; j < 4; j++)
                line( input_img, rect_points[j], rect_points[(j+1)%4], Scalar(0,0,255),2,8);*/
            
            //char text[50];
            
            /*
            // Intersection between blimp's contour and human's contour
            // To avoid including blimp as human
            Mat intersect = Mat::zeros(img_gray.rows, img_gray.cols, CV_8U);
            Mat blimp_mask = Mat::zeros(img_gray.rows, img_gray.cols, CV_8U);
            Mat foreground_mask = Mat::zeros(img_gray.rows, img_gray.cols, CV_8U);
            if (blimp_contour_idx >= 0)
                drawContours(blimp_mask, contours_blimp, blimp_contour_idx, Scalar(255), CV_FILLED);
            drawContours(foreground_mask, contours_foreground, i, Scalar(255), CV_FILLED);
            intersect = blimp_mask & foreground_mask;
            vector<vector<Point> > contours_intersect;
            findContours(intersect.clone(), contours_intersect, CV_RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
            double intersect_area = 0;
            if (contours_intersect.size()) {
                for (int j = 0; j < contours_intersect.size(); j++) {
                    intersect_area += contourArea(contours_intersect[j]);
                }
            }
            //morphologyEx(foreground_mask, foreground_mask, MORPH_DILATE, Mat::ones(3,3,CV_8U), Point(-1,-1), 1);
            
            if (intersect_area < 0.4*area) {
                // The overlap of the foreground blob is less than 40% of the blimp (now arbitrary # TODO get better number
                //Moments m = moments(contours_foreground[i]);
                if (fabs(rect.center.x - u0) + fabs(rect.center.y - v0) < 20) {
                    // Around the center, the orientation of the ellipse can be in any direction, depending on the direction the person is looking to
                    // TODO
                    point.x = rect.center.x;
                    point.y = rect.center.y;
                    point.z = 0.f;
                    //point.z = (float)membershipValue;
                    detected_points.points.push_back(point);
                }
                else {
                    double angle, diff_angle, azimuth_angle, height, width;
                    azimuth_angle = atan((rect.center.y-v0)/(rect.center.x-u0))*180.0/PI;
                    
                    if(rect.size.width < rect.size.height) {
                        //angle = acos(fabs(((rect.center.x-u0)*cos((rect.angle-90.0)*PI/180.0) + (rect.center.y-v0)*sin((rect.angle-90.0)*PI/180.0))/sqrt(std::pow(rect.center.x-u0,2) + std::pow(rect.center.y-v0,2)))) * 180.0/PI;
                        angle = rect.angle;
                        if (angle < 0.0)
                            angle += 90.0;
                        else
                            angle -= 90.0;
                        height = rect.size.height;
                        width = rect.size.width;
                    }
                    else {
                        //angle = acos(fabs(((rect.center.x-u0)*cos(rect.angle*PI/180.0) + (rect.center.y-v0)*sin(rect.angle*PI/180.0))/sqrt(std::pow(rect.center.x-u0,2) + std::pow(rect.center.y-v0,2)))) * 180.0/PI;
                        angle = rect.angle;
                        height = rect.size.width;
                        width = rect.size.height;
                    }
                    diff_angle = angle - azimuth_angle;
                    if (diff_angle > 150.0)
                        diff_angle -= 180.0;
                    else if (diff_angle < -150.0)
                        diff_angle += 180.0;
                    
                    // Writing on image for debug
                    sprintf(text, "%.2lf", diff_angle);
                    putText(cv_ptr->image, text, rect.center, FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0,255,0),2);
                    sprintf(text, "%.2lf %.2lf", rect.angle, rect.size.width/rect.size.height);
                    putText(cv_ptr->image, text, Point(rect.center.x, rect.center.y+30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0,255,255),2);
                    sprintf(text, "%.2lf", atan((rect.center.y-v0)/(rect.center.x-u0))*180.0/PI);
                    putText(cv_ptr->image, text, Point(rect.center.x, rect.center.y+60), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(127,255,0),2);
                    //
                        
                    if (fabs(diff_angle) < 30.0) {
                        // orientation less than 15 degree from the radial direction -- supposed to be human
                        Point2f head_center = Point(rect.center.x + 3.*height/8.*sgn(rect.center.x-u0)*cos(fabs(angle)*PI/180.), rect.center.y + 3.*height/8.*sgn(rect.center.y-v0)*sin(fabs(angle)*PI/180.));
                        RotatedRect ROI(head_center, Size(height/4., 3.*width/4.), angle);
                        //Point2f rect_points[4];
                        ROI.points(rect_points);
                        Point points[4];
                        for(int j = 0; j < 4; j++) {
                            line( cv_ptr->image, rect_points[j], rect_points[(j+1)%4], Scalar(255,255,255),2,8);
                            points[j] = rect_points[j];
                        }
                        Mat temp_mask = Mat::zeros(img_gray.rows, img_gray.cols, CV_8U);
                        Rect ROI_rect = ROI.boundingRect();
                        Rect head_matrix_bound(Point(std::max(0,ROI_rect.x), max(0,ROI_rect.y)), Point(std::min(img_gray.cols, ROI_rect.x+ROI_rect.width), std::min(img_gray.cols, ROI_rect.y+ROI_rect.height)));
                        //rectangle(temp_mask, head_matrix_bound, Scalar(255), -1);
                        fillConvexPoly(temp_mask, points, 4, Scalar(255));
                        rectangle(cv_ptr->image, head_matrix_bound, Scalar(255), 1);
                        temp_mask = temp_mask & foreground_mask;
                        float head_area = sum(temp_mask)[0]/255.0;
                        Mat temp_head(original_img, head_matrix_bound);
                        Mat temp_head_hsv;
                        img_hsv.copyTo(temp_head_hsv, temp_mask);
                        Mat head_hsv(temp_head_hsv, head_matrix_bound);
                        //img_hsv.copyTo(temp_head_hsv, temp_mask);
                        Mat face_mat;//, hair_mat;
                        drawContours(cv_ptr->image, contours_foreground, i, Scalar(0,255,0), 2, CV_AA);        // Draw in green
                        inRange(head_hsv, Scalar(iLowH_skin, iLowS_skin, iLowV_skin), Scalar(iHighH_skin, iHighS_skin, iHighV_skin), face_mat); //Threshold the image, skin
                        //inRange(head_hsv, Scalar(0, 2, 2), Scalar(180, 180, 80), hair_mat); //Threshold the image, hair
                        morphologyEx(face_mat, face_mat, MORPH_CLOSE, Mat::ones(3,3,CV_8U), Point(-1,-1), 1);
                        //morphologyEx(hair_mat, hair_mat, MORPH_CLOSE, Mat::ones(3,3,CV_8U), Point(-1,-1), 1);
                        
                        //Point2f face_center;//, hair_center;
                        bool face_found = false;
                        double face_area;//, hair_area;
                        
                        vector<vector<Point> > contours;
                        findContours(face_mat.clone(), contours, CV_RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
                        if (contours.size() > 0) {
                            std::sort(contours.begin(), contours.end(), compareContourAreas);
                            //Moments mu = moments(contours[0], true);
                            //face_center = Point(mu.m10/mu.m00, mu.m01/mu.m00);
                            //circle(cv_ptr->image, face_center+Point2f(ROI.boundingRect().x,ROI.boundingRect().y), 4, Scalar(255,255,255));
                            
                            face_area = contourArea(contours[0]);
                            //sprintf(text, "%.4f, %.4f", face_area, head_area);
                            //putText(cv_ptr->image, text, head_center, FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255,255,255) ,2);
                            Mat face_show;
                            temp_head.copyTo(face_show, face_mat);
                            imshow("Face", face_show);
                            if (face_area >= 0.4*head_area) {
                                // Face is large enough -- half of the head
                                face_found = true;
                            }
                        }
                    }
                    else {
                        drawContours(cv_ptr->image, contours_foreground, i, Scalar(0,0,255), 1, CV_AA);        // Draw in red
                        sprintf(text, "%.2lf", diff_angle);
                        putText(cv_ptr->image, text, rect.center, FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0,255,0),2);
                        sprintf(text, "%.2lf %.2lf", rect.angle, rect.size.width/rect.size.height);
                        putText(cv_ptr->image, text, Point(rect.center.x, rect.center.y+30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0,255,255),2);
                        sprintf(text, "%.2lf", azimuth_angle);
                        putText(cv_ptr->image, text, Point(rect.center.x, rect.center.y+60), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(127,255,0),2);
                    }
                }
                
            }
            else {
                // Supposed to be blimp. Draw for debug
                drawContours(cv_ptr->image, contours_foreground, i, Scalar(255,0,0), 2, CV_AA);        // Draw in blue
            }*/
    if (save_video) {           // TODO If save
        f << count_img << ",";
        f << tracked_objects.size() << "," << tracked_humans.size() << "," << objects.size() << "," << humans.size() << "," << heads.size() << ",";
        for (int a = 0; a < tracked_objects.size(); a++) {
            f << tracked_objects[a].getStringForSave() << ",";
        }
        for (int a = 0; a < tracked_humans.size(); a++) {
            f << tracked_humans[a].getStringForSave() << ",";
        }
        for (int a = 0; a < objects.size(); a++) {
            RotatedRect temp = objects[a];
            f << temp.center.x << "," << temp.center.y << "," << temp.size.width << "," << temp.size.height << "," << temp.angle << ",";
        }
        for (int a = 0; a < humans.size(); a++) {
            RotatedRect temp = humans[a];
            f << temp.center.x << "," << temp.center.y << "," << temp.size.width << "," << temp.size.height << "," << temp.angle << ",";
        }
        for (int a = 0; a < heads.size(); a++) {
            RotatedRect temp = heads[a];
            f << temp.center.x << "," << temp.center.y << "," << temp.size.width << "," << temp.size.height << "," << temp.angle << ",";
        }
        f << double(total_time)/getTickFrequency() * 1000. << endl;         // millisecond
    }
            Mat objMat, detectedMat;
            input_img.copyTo(objMat);
            input_img.copyTo(detectedMat);
        if (toDraw) {
	        //for(int i = 0; i < contours_foreground.size(); i++){
	        //	drawContours(input_img, contours_foreground, i, Scalar(0,255,0), 1, CV_AA);
	        //}
	        for (int i = 0; i < humans.size(); i++) {
		        Point2f rect_points[4];
		        humans[i].points(rect_points);
		        for(int j = 0; j < 4; j++) {
			        line( detectedMat, rect_points[j], rect_points[(j+1)%4], Scalar(255,255,0),2,8);
			        line( input_img, rect_points[j], rect_points[(j+1)%4], Scalar(255,255,0),2,8);
		        }
	        }
	        for (int i = 0; i < heads.size(); i++) {
		        Point2f rect_points[4];
		        heads[i].points(rect_points);
		        for(int j = 0; j < 4; j++) {
			        line( detectedMat, rect_points[j], rect_points[(j+1)%4], Scalar(0,255,255),2,8);
			        line( input_img, rect_points[j], rect_points[(j+1)%4], Scalar(0,255,255),2,8);
		        }
	        }
	        for (int i = 0; i < objects.size(); i++) {
		        Point2f rect_points[4];
		        objects[i].points(rect_points);
		        for(int j = 0; j < 4; j++) {
			        line( objMat, rect_points[j], rect_points[(j+1)%4], Scalar(0,0,255),2,8);
			        line( input_img, rect_points[j], rect_points[(j+1)%4], Scalar(0,0,255),2,8);
		        }
		        rawBoxes[i].points(rect_points);
		        for(int j = 0; j < 4; j++)
			        line( objMat, rect_points[j], rect_points[(j+1)%4], Scalar(255,255,0),1,8);
                area_heads[i].points(rect_points);
                for(int j = 0; j < 4; j++)
                    line( objMat, rect_points[j], rect_points[(j+1)%4], Scalar(0,255,0),2,8);
	        }

	        for (int track = 0; track < tracked_objects.size(); track++) {
		        Scalar color(255,255,255);
		        TrackedObject object = tracked_objects[track];
		        if (object.getCount() == 0) {
			        circle(input_img, object.getPointBody(), 2, color, -1);
			        continue;
		        }
		        circle(input_img, object.getPointBody(), 2, color, -1);
		        Point2f rect_points[4];
		        object.getBodyROI().points(rect_points);
		        for(int j = 0; j < 4; j++)
			        line( input_img, rect_points[j], rect_points[(j+1)%4], Scalar(192,192,0),1,8);
		        //circle(input_img, object.getPointBody(), 3*object.getSdBody(), Scalar(192,192,0), 2);

		        //circle(input_img, object.getPointHead(), 2, Scalar(192,192,192), -1);
		        object.getHeadROI().points(rect_points);
		        for(int j = 0; j < 4; j++)
			        line( input_img, rect_points[j], rect_points[(j+1)%4], Scalar(255,255,255),1,8);
		        //circle(input_img, object.getPointHead(), 3*object.getSdHead(), Scalar(255,0,0), 1);
	        }
	        
	        if (blimp_bb.center != Point2f(0.,0.) || blimp_bb.size != Size2f(0.,0.) || blimp_bb.angle != 0.) {
	            Point2f rect_points[4];
		        blimp_bb.points(rect_points);
		        for(int j = 0; j < 4; j++)
			        line( input_img, rect_points[j], rect_points[(j+1)%4], Scalar(255,0,0),2,8);
		        drawContours(input_img, contours_blimp, 0, Scalar(255,64,0), 1);
	        }

	        for (int track = 0; track < tracked_humans.size(); track++) {
		        Scalar color(0,255,0);
		        TrackedObject human = tracked_humans[track];
		        circle(input_img, human.getPointBody(), 2, color, -1);
		        Point2f rect_points[4];
		        human.getBodyROI().points(rect_points);
		        for(int j = 0; j < 4; j++)
			        line( input_img, rect_points[j], rect_points[(j+1)%4], Scalar(192,192,0),2,8);
		        //circle(input_img, object.getPointBody(), 3*object.getSdBody(), Scalar(192,192,0), 2);

		        //circle(input_img, object.getPointHead(), 2, Scalar(192,192,192), -1);
		        human.getHeadROI().points(rect_points);
		        for(int j = 0; j < 4; j++)
			        line( input_img, rect_points[j], rect_points[(j+1)%4], Scalar(64,223,0),2,8);
		        //circle(input_img, object.getPointHead(), 3*object.getSdHead(), Scalar(255,0,0), 1);
		        int dir = human.getDirection();
		        float angle = (dir - human.getHeadROI().angle)*CV_PI/180.;
		        arrowedLine(input_img, human.getPointHead(), human.getPointHead() + 50.*Point2f(sin(angle), cos(angle)), Scalar(64,223,0), 2);
		        char buffer[10];
                //arrowedLine(input_img, human.getPointHead(), human.getPointHead() + 10.*human.getHeadVel(), Scalar(0,0,255), 1);
		        //sprintf(buffer, "%d, %d", angles[track], human.getDirection());
		        //putText(input_img, buffer , human.getBodyROI().center+50.*Point2f(-sin(human.getHeadROI().angle*CV_PI/180.),cos(human.getHeadROI().angle*CV_PI/180.)), FONT_HERSHEY_PLAIN, 1, Scalar(255,0,255), 2);
	        }

	        //imshow("FG Mask MOG 2", fgMaskMOG2);
	        //Mat resized;
	        //resize(input_img, resized, Size(500,500));
	        //imshow("Detection", resized);
        }
        if(save_video) {
	        outputVideo << input_img;
            //ObjVideo << objMat;
            //DetectedVideo << detectedMat;
        }
        //waitKey(1);
        //std::cout << end-begin << std::endl;
        count_img++;
        return (tracked_humans.size() > 0 || tracked_objects.size() > 0);
}

void BGSub::groupContours ( vector< vector<Point> > inputContours, vector<RotatedRect> &outputBoundingBoxes, vector<RotatedRect> &rawBoundingBoxes, double distanceThreshold ) {
    if (!inputContours.size())
        return;
    // inputContours should be sorted in descending area order
    outputBoundingBoxes.clear();
    rawBoundingBoxes.clear();
    //int j;
    for (vector< vector<Point> >::iterator it = inputContours.begin(); it != inputContours.end(); ) {
        if (contourArea(*it) < area_threshold)          // Too small to be the seed
            break;
        vector<Point> contour_i = *it;
        RotatedRect rect_i = minAreaRect(contour_i);
        Point2f center_i = rect_i.center;
        if (norm(center_i - img_center) > img_center.x - imgBorder - 30) {            // TODO Now HARDCODED for blimp detection
            // Preventing some residue from blimp's color detection
            it = inputContours.erase(it);
            continue;
        }
        Size h_size = getHumanSize(norm(center_i - img_center));
        vector< vector<Point> >::iterator it_j = it+1;
        while (it_j != inputContours.end()) {
            vector<Point> contour_j = *it_j;
            RotatedRect rect_j = minAreaRect(contour_j);
            Point2f center_j = rect_j.center;
            if (norm(center_j - img_center) > img_center.x - imgBorder - 30) {        // TODO Now HARDCODED for blimp detection
                // Preventing some residue from blimp's color detection
                it_j = inputContours.erase(it_j);
                continue;
            }
            double d_ij = norm(center_i - center_j);        // Distance between 2 contours
            if (d_ij < h_size.height) {
                contour_i.insert(contour_i.end(), contour_j.begin(), contour_j.end());
                rect_i = minAreaRect(contour_i);
                center_i = rect_i.center;
                h_size = getHumanSize(norm(center_i-img_center));
                it_j = inputContours.erase(it_j);
            }
            else {
                ++it_j;
            }
        }

        //if (contourArea(contour_i) < area_threshold) {
        //    continue;
        //}
            
        RotatedRect rect = minAreaRect(contour_i);
        Point2f center = rect.center;
        float w = rect.size.width;
        float h = rect.size.height;
        float phi = rect.angle;
        rawBoundingBoxes.push_back(rect);
        float theta = atan2(center.x - img_center.x, img_center.y - center.y) *180./CV_PI;
        if (w <= h) {
            if (phi - theta > 90.)
                phi -= 180.;
            else if (phi - theta < -90.)
                phi += 180.;
        }
        else {
            float temp = w;
            w = h;
            h = temp;
            if (phi - theta > 0.)
                phi -= 90.;
            else if (phi - theta < -180.)
                phi += 270;
            else
                phi += 90.;
        }
        float delta = abs(phi - theta) * CV_PI/180.;
        if (delta > CV_PI/2.) {            // width < height --> 90 deg change
            float temp = w;
            w = h;
            h = temp;
            delta -= CV_PI/2.;
        }
        float w_aligned = h*sin(delta) + w*cos(delta);
        w_aligned *= 1.5;
        float h_aligned = h*cos(delta) + w*sin(delta);
        h_aligned *= 1.5;

        //Size human_size = getHumanSize(norm(center - img_center)) + Size(10,20);
        Size human_size = getHumanSize(norm(center - img_center));
        human_size.width = cvRound(1.5*human_size.width);
        human_size.height = 2*human_size.width;
        outputBoundingBoxes.push_back(RotatedRect(center, Size(max(w_aligned,float(human_size.width)), max(h_aligned,float(human_size.height))), theta));
        ++it;
    }
    if (outputBoundingBoxes.size() > 1) {
        vector<RotatedRect>::iterator it_bb = outputBoundingBoxes.begin(), it_bb2;
        vector< vector<Point> >::iterator it_contour = inputContours.begin(), it_contour2;
        for ( ; it_bb != outputBoundingBoxes.end()-1; ) {
            if (it_contour == inputContours.end() - 1) {
                std::cout << "Something wrong!" << std::endl;
                break;
                // Should not happen anyway
            }
            it_bb2 = it_bb + 1;
            it_contour2 = it_contour + 1;
            RotatedRect r1 = *it_bb;
            float area1 = r1.size.area();
            vector<Point> contour_i = *it_contour;
            for ( ; it_bb2 != outputBoundingBoxes.end(); ) {
                if (it_contour2 == inputContours.end()) {
                    std::cout << "Something wrong 2!" << std::endl;
                    break;
                    // Should not happen anyway
                }
                RotatedRect r2 = *(it_bb2);
                float area2 = r2.size.area();
                vector<Point> contour_j = *it_contour2;

                vector<Point2f> v, hull;
                int ret = rotatedRectangleIntersection(r1,r2,v);
                if (ret == INTERSECT_FULL) {
                    rawBoundingBoxes.erase(rawBoundingBoxes.begin() + (it_bb2 - outputBoundingBoxes.begin()));
                    outputBoundingBoxes.erase(it_bb2);
                    it_bb2 = it_bb + 1;
                    inputContours.erase(it_contour2);
                    it_contour2 = it_contour + 1;
                }
                else if (ret == INTERSECT_PARTIAL){
                    float intArea;
                    convexHull(v, hull);
                    intArea = contourArea(hull);
                    if (intArea/fmin(area1, area2) > 0.5) {
                        contour_i.insert(contour_i.end(), contour_j.begin(), contour_j.end());
                        RotatedRect rect = minAreaRect(contour_i);
                        Point2f center = rect.center;
                        float w = rect.size.width;
                        float h = rect.size.height;
                        float phi = rect.angle;
                        rawBoundingBoxes[it_bb-outputBoundingBoxes.begin()] = rect;
                        float theta = atan2(center.x - img_center.x, img_center.y - center.y) *180./CV_PI;
                        if (w <= h) {
                            if (phi - theta > 90.)
                                phi -= 180.;
                            else if (phi - theta < -90.)
                                phi += 180.;
                        }
                        else {
                            float temp = w;
                            w = h;
                            h = temp;
                            if (phi - theta > 0.)
                                phi -= 90.;
                            else if (phi - theta < -180.)
                                phi += 270;
                            else
                                phi += 90.;
                        }
                        float delta = abs(phi - theta) * CV_PI/180.;
                        if (delta > CV_PI/2.) {            // width < height --> 90 deg change
                            float temp = w;
                            w = h;
                            h = temp;
                            delta -= CV_PI/2.;
                        }
                        float w_aligned = h*sin(delta) + w*cos(delta);
                        w_aligned *= 1.5;
                        float h_aligned = h*cos(delta) + w*sin(delta);
                        h_aligned *= 1.5;

                        //Size human_size = getHumanSize(norm(center - img_center)) + Size(10,20);
                        Size human_size = getHumanSize(norm(center - img_center));
                        human_size.width = cvRound(1.5*human_size.width);
                        human_size.height = 2*human_size.width;
                        r1 = RotatedRect(center, Size(max(w_aligned,float(human_size.width)), max(h_aligned,float(human_size.height))), theta);
                        *it_bb = r1;
                        area1 = r1.size.area();
                        rawBoundingBoxes.erase(rawBoundingBoxes.begin() + (it_bb2 - outputBoundingBoxes.begin()));
                        outputBoundingBoxes.erase(it_bb2);
                        it_bb2 = it_bb + 1;
                        inputContours.erase(it_contour2);
                        it_contour2 = it_contour + 1;
                    }
                    else {
                        it_bb2++;
                        it_contour2++;
                    }
                }
                else {
                    it_bb2++;
                    it_contour2++;
                }
            }
            it_bb++;
            it_contour++;
            if (outputBoundingBoxes.size() == 1 || it_bb == outputBoundingBoxes.end())
                break;
        }
    }
}

RotatedRect BGSub::groupBlimp ( vector< vector<Point> > &inputContours, double distanceThreshold ) {
    if (!inputContours.size())
        return RotatedRect(Point2f(0.,0.), Size2f(0.,0.),0.);

    if (inputContours.size() > 1) {
        if (contourArea(inputContours[0]) < 100.)
            return RotatedRect(Point2f(0.,0.), Size2f(0.,0.),0.);
        vector< vector<Point> >::iterator it_contour = inputContours.begin(), it_contour2;
        it_contour2 = it_contour + 1;
        RotatedRect r1 = minAreaRect(*it_contour);
        float area1 = r1.size.area();
        for ( ; it_contour2 != inputContours.end(); ) {
            RotatedRect r2 = minAreaRect(*it_contour2);
            vector<Point> contour_j = *it_contour2;
            float area2 = r2.size.area();

            vector<Point2f> v, hull;
            int ret = rotatedRectangleIntersection(r1,r2,v);
            if (ret != INTERSECT_NONE) {
                it_contour->insert(it_contour->end(), contour_j.begin(), contour_j.end());
                // update bounding box
                r1 = minAreaRect(*it_contour);
                inputContours.erase(it_contour2);
                it_contour2 = it_contour + 1;
            }
            else {
                it_contour2++;
            }
        }
        return r1;
    }
    else {
        return minAreaRect(inputContours[0]);
    }
}


Size BGSub::getHumanSize(float radius) {
    if (radius < 0.1) {
        return Size(100.f, 200.f);
    }
	float width;

	width = cvRound(humanWidth * radius /((camHeight-humanHeight) * tan(radius/(m*k1))));
	width = max(32.f, min(100.f, width));
	/*if (radius > 280)
		width = 24.;
	else if (radius < 120)
		width = 88.;
	else
		width = cvRound(136.26 - 0.4*radius);*/
	return Size(width, 2*width);
}

void BGSub::detectOriginalHOG(Mat &img, vector<RotatedRect> &ROIs, vector<RotatedRect> &detections, Size size_min, Size size_max, double scale0, int flag) {
    // flag == 0 --> body
    // flag == 1 --> head
    if (useFisheyeHOG)
        return;             // Should not even enter this

    //cout << "\tStart" << endl;
    detections.clear();
    vector<double> weights;
    RotatedRect area;
    Point2f vertices[4];
    Point2f img_center(float(img.cols/2), float(img.rows/2));
    float r1, r2, theta1, theta2, width1, width2;
    vector<Point3f> limits;
    Size win_size;
    if (flag == 0) {
        win_size = hog_body_orig.winSize;
    }
    else if (flag == 1) {
        win_size = hog_head_orig.winSize;
    }

    // resize for all sizes
    double scale = double(size_min.width) / double(win_size.width);
    //Size maxSz(cvCeil(img.cols/scale), cvCeil(img.rows/scale));
    int levels = 0;
    //Mat smallerImgBuf(maxSz, img.type());
    vector<Mat> resized_imgs;

    vector<double> levelScale;
    for( levels = 0; levels < 64; levels++ )
    {
        Size sz(cvRound(img.cols/scale), cvRound(img.rows/scale));
        Mat scaled_img(sz, img.type());
        if( sz == img.size() )
            scaled_img = Mat(sz, img.type(), img.data, img.step);
        else
            resize(img, scaled_img, sz);
        resized_imgs.push_back(scaled_img);
        levelScale.push_back(scale);
        if( cvRound(img.cols/(2*scale) - imgBorder) < win_size.width ||
            cvRound(img.rows/(2*scale) - imgBorder) < win_size.height ||
            scale0 <= 1 ||
            scale*win_size.width > size_max.width)
            break;
        scale *= scale0;
    }

    for (int a = 0; a < ROIs.size(); a++) {
        area = ROIs[a];
        area.points(vertices);
        r1 = norm(area.center-img_center) - area.size.height/2;
        r2 = r1 + area.size.height;
        width1 = area.size.width;
        width2 = area.size.width/2.;
        // r1 must be less than r2
        CV_Assert(r1 < r2);
        if (r1 < 0) {               // center is inside the area
            r1 = 0.;
            theta1 = area.angle - 89.;      // Avoiding cos = 0
            while (theta1 < 0)
                theta1 += 360.;
            theta2 = area.angle + 89.;
            while (theta2 < 0)
                theta2 += 360.;
            //cout << "r1: " << r1 << ", " << " thetas: " << theta1 << "," << theta2 << endl;
        }
        else {
            theta1 = atan2(vertices[0].x - img_center.x, img_center.y - vertices[0].y) * 180./CV_PI;

            // convert to [0,360) range
            if (theta1 < 0)
                theta1 += 360.;
            theta2 = atan2(vertices[3].x - img_center.x, img_center.y - vertices[3].y) * 180./CV_PI;
            if (theta2 < 0)
                theta2 += 360.;
        }

        limits.push_back(Point3f(r1, theta1, width1));
        limits.push_back(Point3f(r2, theta2, width2));
    }

    float angle_step = 2.;      // 2 degree
    Point3f lim1, lim2;
    for (float angle = 0.; angle < 360.; angle += angle_step) {
        // First screening
        bool isOutOfBound = true;
        vector<int> matchedROI;
        for (int check = 0; check < limits.size(); check += 2) {
            lim1 = limits[check];
            lim2 = limits[check+1];
            if (lim1.y > lim2.y) {                  // theta1 > theta2 : crossing the 0-degree line
                if (angle > lim1.y || angle < lim2.y) {
                    isOutOfBound = false;
                    matchedROI.push_back(check);
                }
            }
            else if (lim1.x == 0) {             // area covering image's center
                if (angle > lim1.y && angle < lim2.y) {
                    isOutOfBound = false;
                    matchedROI.push_back(check);
                }
            }
            else {                                  // normal
                if (angle > lim1.y && angle < lim2.y) {
                    isOutOfBound = false;
                    matchedROI.push_back(check);
                }
            }
        }
        if (isOutOfBound)
            continue;

        //cout << "In bound " << angle << endl;

        float r1 = limits[matchedROI[0]].x;
        float r2 = limits[matchedROI[0]+1].x;
        float r_min, r_max;

        float center_angle = ROIs[matchedROI[0]/2].angle;
        if (center_angle < 0)
            center_angle += 360.;

        r_min = (r1/ cos((angle - center_angle)*CV_PI/180.));

        if (r2 * fabs(tan((angle - center_angle)*CV_PI/180.)) < limits[matchedROI[0]+1].z) {
            r_max = (r2/ cos((angle - center_angle)*CV_PI/180.));
        }
        else {
            r_max = (limits[matchedROI[0]+1].z/ fabs(sin((angle - center_angle)*CV_PI/180.)));
        }
        //cout << "\t" << angle << "," << center_angle << "---";
        //float r_min = 1./scale*limits[matchedROI[0]].x;
        //float r_max = 1./scale*limits[matchedROI[0]+1].x;
        //float width = win_size.width / scale;
        Rect_<float> crop_rect(img.cols/2 - win_size.width/2, img.rows/2 - r_max, win_size.width, r_max-r_min);
        vector<Rect_<float> > rects;
        rects.push_back(crop_rect);

        for (int m = 1; m < matchedROI.size(); m++) {
            r1 = limits[matchedROI[m]].x;
            r2 = limits[matchedROI[m]+1].x;
            float new_min, new_max;
            center_angle = ROIs[matchedROI[m]/2].angle;
            if (center_angle < 0)
                center_angle += 360.;
            new_min = (r1/ cos((angle - center_angle)*CV_PI/180.));
            if (r2 * fabs(tan((angle - center_angle)*CV_PI/180.)) < limits[matchedROI[m]+1].z)
                new_max = (r2/ cos((angle - center_angle)*CV_PI/180.));
            else {
                new_max = (limits[matchedROI[m]+1].z/ fabs(sin((angle - center_angle)*CV_PI/180.)));
            }
            Rect_<float> new_rect(img.cols/2 - win_size.width/2, img.rows/2 - new_max, win_size.width, new_max-new_min);
            int r = 0;
            for (; r < rects.size(); r++) {
                Rect_<float> overlap = rects[r] | new_rect;
                Rect_<float> intersect = rects[r] & new_rect;
                if (intersect.area() > 0) {
                    rects[r] = overlap;
                    break;
                }
            }
            if (r == rects.size()) {
                // no overlap yet
                rects.push_back(new_rect);      // Add another rect
            }
        }

        // Recheck for new overlap
        bool new_update = true;
        while (new_update) {
            new_update = false;
            for (vector<Rect_<float> >::iterator it = rects.begin(); it != rects.end()-1; ) {
                vector<Rect_<float> >::iterator it_j = it+1;
                for (; it_j != rects.end(); ) {
                    Rect_<float> rect_j = *it_j;
                    Rect_<float> intersect = (*it) & rect_j;
                    if (intersect.area() > 0 ) {
                        *it = (*it) | rect_j;
                        it_j = rects.erase(it_j);
                        new_update = true;
                    }
                    else {
                        it_j++;
                    }
                }
                it++;
                if (it == rects.end())
                    break;
            }
        }

        for (int s = 0; s < levelScale.size(); s++) {
            double scale = levelScale[s];
            Mat scaled_img = resized_imgs[s];
            Mat M = getRotationMatrix2D(Point2f(scaled_img.cols/2, scaled_img.rows/2), angle, 1.0);
            Mat rotated, cropped;
            warpAffine(scaled_img, rotated, M, scaled_img.size(), INTER_CUBIC);

            for (int c = 0; c < rects.size(); c++) {
                if (cvRound(rects[c].height/scale) < win_size.height)
                    continue;
                Size crop_size(cvRound(rects[c].width), cvRound(rects[c].height/scale));
                getRectSubPix(rotated, crop_size, 0.5/scale*(rects[c].tl()+rects[c].br()), cropped);
                vector<Point> foundLocations;
                vector<double> foundWeights;
                if (flag == 0) {
                    hog_body_orig.detect(cropped, foundLocations, foundWeights, 0., Size(4,4), Size(0,0));
                }
                else if (flag == 1) {
                    hog_head_orig.detect(cropped, foundLocations, foundWeights, 8.3, Size(4,4), Size(0,0));
                }

                Size2f scaledWinSize(scale*win_size.width, scale*win_size.height);
                Point2f centerOffset(0.5*win_size.width, 0.5*win_size.height);
                Point2f imgCenter = Point2f(scaled_img.cols/2, scaled_img.rows/2);
                for( size_t j = 0; j < foundLocations.size(); j++ )
                {
                    Point2f topLeft(foundLocations[j].x, foundLocations[j].y);
                    Point2f center = topLeft + centerOffset + 1./scale*rects[c].tl();
                    float R = norm(center - imgCenter)*scale;

                    detections.push_back(RotatedRect(R*Point2f(sin(angle*CV_PI/180.), -cos(angle*CV_PI/180.)) + img_center,
                                                     scaledWinSize, angle));
                    weights.push_back(foundWeights[j]);
                }
            }
        }
    }
    groupRectanglesNMS(detections, weights, 1, 0.4);
    //cout << "\tEnd" << endl;
    return;
}

void BGSub::groupRectanglesNMS(vector<cv::RotatedRect>& rectList, vector<double>& weights, int groupThreshold, double overlapThreshold) const
{
    if( groupThreshold <= 0 || overlapThreshold <= 0 || rectList.empty() )
    {
        return;
    }

    CV_Assert(rectList.size() == weights.size());

    // Sort the bounding boxes by the detection score
    std::multimap<double, size_t> idxs;
    for (size_t i = 0; i < rectList.size(); ++i)
    {
        idxs.insert(std::pair<double, size_t>(weights[i], i));
    }

    vector<RotatedRect> outRects;
    vector<double> outWeights;

    // keep looping while some indexes still remain in the indexes list
    while (idxs.size() > 0)
    {
        // grab the last rectangle
        std::multimap<double, size_t>::iterator lastElem = --idxs.end();
        const cv::RotatedRect& rect1 = rectList[lastElem->second];

        int neigborsCount = 0;
        float scoresSum = lastElem->first;

        idxs.erase(lastElem);

        vector<Point2f> vers, hull;

        for (std::multimap<double, size_t>::iterator pos = idxs.begin(); pos != idxs.end(); )
        {
            // grab the current rectangle
            const cv::RotatedRect& rect2 = rectList[pos->second];

            int ret = rotatedRectangleIntersection(rect1,rect2,vers);
            float intArea;
            if (ret != INTERSECT_NONE) {
                convexHull(vers, hull);
                intArea = contourArea(hull);
            }
            else
                intArea = 0.;
            float area = min(rect1.size.area(), rect2.size.area());
            float overlap = intArea / area;

            // if there is sufficient overlap, suppress the current bounding box
            if (overlap > overlapThreshold)
            {
                scoresSum += pos->first;
                std::multimap<double, size_t>::iterator save = pos;
                ++save;
                idxs.erase(pos);
                pos = save;
                ++neigborsCount;
            }
            else
            {
                ++pos;
            }
        }
        if (neigborsCount >= groupThreshold)
        {
            outRects.push_back(rect1);
            outWeights.push_back(scoresSum);
        }
    }
    rectList = outRects;
    weights = outWeights;
}
