#include "BGSub.h"
#include "TrackedObject.h"
#include <opencv2/opencv.hpp>
#include <opencv/highgui.h>
#include <math.h>
#include <iostream>
#include "fern_based_classifier.h"
#include "BGSub.h"

using namespace cv;
using namespace std;

bool compareContourAreas ( vector<Point> contour1, vector<Point> contour2 ) {
    double i = contourArea(Mat(contour1));
    double j = contourArea(Mat(contour2));
    return ( i > j );
}

BGSub::BGSub (bool _toDraw, bool _toSave) {
    area_threshold = 30;
    
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
		string videoName = "omni1A_test1_bgslib_improved.avi";
		outputVideo.open(videoName, CV_FOURCC('D','I','V','X'), 10, Size(800, 600), true);
		//outputVideo.open(videoName, CV_FOURCC('D','I','V','X'), 10, Size(768, 768), true);
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

    char classifier_name[] = "classifiers/classifier_acc_400-4";
	classifier = new fern_based_classifier(classifier_name);

	hog_size = classifier->hog_image_size;

    hog_body.load("/home/veerachart/HOG_Classifiers/32x64_weighted/cvHOGClassifier_32x64+hard.yaml");
    hog_head.load("/home/veerachart/HOG_Classifiers/head_fastHOG.yaml");
    hog_direction = FisheyeHOGDescriptor(Size(hog_size,hog_size), Size(hog_size/2,hog_size/2), Size(hog_size/4,hog_size/4), Size(hog_size/4,hog_size/4), 9);
    hog_original = HOGDescriptor(Size(hog_size,hog_size), Size(hog_size/2,hog_size/2), Size(hog_size/4,hog_size/4), Size(hog_size/4,hog_size/4), 9);
}

BGSub::~BGSub() {
	delete classifier;
}

bool BGSub::processImage (Mat &input_img) {
    if (img_center == Point2f() )
        img_center = Point2f(input_img.cols/2, input_img.rows/2);
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
    threshold(fgMaskMOG2, fgMaskMOG2, 128, 255, THRESH_BINARY);
    //outputVideo << save;
    
    Mat intersect = img_thresholded_b & fgMaskMOG2;         // Blimp
    fgMaskMOG2 -= intersect;                                // Human
    
    morphologyEx(intersect, intersect, MORPH_OPEN, Mat::ones(3,3,CV_8U));
    morphologyEx(intersect, intersect, MORPH_CLOSE, Mat::ones(5,5,CV_8U));

    vector<vector<Point> > contours_blimp;
    findContours(intersect, contours_blimp, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
    
    RotatedRect blimp_bb;
    Mat mask_blimp = Mat::zeros(input_img.size(), CV_8UC1);
    vector< vector < Point2f > > hull(1);
    if (contours_blimp.size()) {
        std::sort(contours_blimp.begin(), contours_blimp.end(), compareContourAreas);
        blimp_bb = groupBlimp(contours_blimp, 1.0);
        blimp_center = blimp_bb.center;
        /*if (blimp_bb.center != Point2f(0.,0.) || blimp_bb.size != Size2f(0.,0.) || blimp_bb.angle != 0.) {
            convexHull(contours_blimp[0],hull[0]);
            drawContours(mask_blimp, hull, 0, Scalar(255), CV_FILLED);
        }*/
    }
    
    //fgMaskHOG2 = fgMaskHOG2 & ~mask_blimp;
                
    morphologyEx(fgMaskMOG2, fgMaskMOG2, MORPH_OPEN, Mat::ones(3,3,CV_8U), Point(-1,-1), 1);
    morphologyEx(fgMaskMOG2, fgMaskMOG2, MORPH_CLOSE, Mat::ones(5,5,CV_8U), Point(-1,-1), 2);
    morphologyEx(fgMaskMOG2, fgMaskMOG2, MORPH_OPEN, Mat::ones(3,3,CV_8U), Point(-1,-1), 1);
    imshow("FG Mask MOG 2", fgMaskMOG2);

    vector<vector<Point> > contours_foreground;
    findContours(fgMaskMOG2.clone(), contours_foreground, CV_RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    vector<RotatedRect> humans;
    vector<RotatedRect> heads;
    vector<double> weights;
    vector<float> descriptors;

    vector<RotatedRect> objects, rawBoxes;
    vector<RotatedRect> area_heads;					// ROI to search for heads = top half of objects

    if(contours_foreground.size() > 0){
        std::sort(contours_foreground.begin(), contours_foreground.end(), compareContourAreas);
        
        double threshold = 1.0;
        groupContours(contours_foreground, objects, rawBoxes, threshold);


        if (objects.size()) {
        	Size size_min(1000,1000), size_max(0,0);
        	for (int obj = 0; obj < objects.size(); obj++) {
        		Size temp = getHumanSize(norm(objects[obj].center - img_center));
        		if (temp.width < size_min.width)
        			size_min = temp;
        		if (temp.width > size_max.width)
        			size_max = temp;

        		float theta_r = objects[obj].angle*CV_PI/180.;
        		area_heads.push_back(RotatedRect(objects[obj].center + 0.25*objects[obj].size.height*Point2f(sin(theta_r), -cos(theta_r)), Size(objects[obj].size.width,objects[obj].size.height/2), objects[obj].angle));
        		//cout << objects[obj].center << " and " << area_heads.back().center << endl;
        	}
        	size_min -= Size(10,20);
        	size_max += Size(10,20);
        	float width_head_min = max(12., 0.375*size_min.width - 10.);
        	Size size_head_min(width_head_min, width_head_min);
        	float width_head_max = max(12., 0.375*size_max.width + 10.);
	        Size size_head_max(width_head_max, width_head_max);

	        //cout << size_min << " " << size_max << " " << size_head_min << " " << size_head_max << endl;

        	hog_body.detectAreaMultiScale(input_img, objects, humans, weights, descriptors, size_min, size_max, 0., Size(4,2), Size(0,0), 1.05, 1.0);

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

        	hog_head.detectAreaMultiScale(input_img, area_heads, heads, weights, descriptors, size_head_min, size_head_max, 8.2, Size(4,2), Size(0,0), 1.05, 1.0);


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
			        if (best_dist < 3*tracked_humans[best_track].getSdHead()) {
				        // Update
				        //cout << "Update head:" << heads[head].center << "," << heads[head].size << " with " << tracked_humans[best_track].getBodyROI().center << "," << tracked_humans[best_track].getBodyROI().size << endl;
				        tracked_humans[best_track].UpdateHead(heads[head]);
				        humanHasHead[best_track] = true;
				        continue;
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

			        if (best_dist < 3*tracked_objects[best_track].getSdHead()) {
				        // Update
				        //cout << "Update head:" << heads[head].center << "," << heads[head].size << " with " << tracked_objects[best_track].getBodyROI().center << "," << tracked_objects[best_track].getBodyROI().size << endl;
				        tracked_objects[best_track].UpdateHead(heads[head]);
				        objectHasHead[best_track] = true;
				        continue;
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
    vector<int> classes, angles;
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
		        Point2f head_vel = it->getHeadVel();
		        RotatedRect rect = it->getHeadROI();

		        float walking_dir;
		        if (norm(head_vel) < 1.5) {				// TODO Threshold adjust
			        walking_dir = -1.;					// Not enough speed, no clue
		        }
		        else {
			        walking_dir = rect.angle + atan2(head_vel.x, head_vel.y)*180./CV_PI;			// Estimated walking direction relative to the radial line (0 degree head direction)
			        if (walking_dir < 0)
				        walking_dir += 360;					// [0, 360) range. Negative means no clue
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
		        vector<RotatedRect> location;
		        location.push_back(rect);
		        hog_direction.compute(original_img, descriptor, location);
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
        if (toDraw) {
	        //for(int i = 0; i < contours_foreground.size(); i++){
	        //	drawContours(input_img, contours_foreground, i, Scalar(0,255,0), 1, CV_AA);
	        //}
	        for (int i = 0; i < humans.size(); i++) {
		        Point2f rect_points[4];
		        humans[i].points(rect_points);
		        for(int j = 0; j < 4; j++)
			        line( input_img, rect_points[j], rect_points[(j+1)%4], Scalar(255,255,0),2,8);
	        }
	        for (int i = 0; i < heads.size(); i++) {
		        Point2f rect_points[4];
		        heads[i].points(rect_points);
		        for(int j = 0; j < 4; j++)
			        line( input_img, rect_points[j], rect_points[(j+1)%4], Scalar(0,255,255),2,8);
	        }
	        for (int i = 0; i < objects.size(); i++) {
		        Point2f rect_points[4];
		        objects[i].points(rect_points);
		        for(int j = 0; j < 4; j++)
			        line( input_img, rect_points[j], rect_points[(j+1)%4], Scalar(0,0,255),2,8);
		        rawBoxes[i].points(rect_points);
		        for(int j = 0; j < 4; j++)
			        line( input_img, rect_points[j], rect_points[(j+1)%4], Scalar(255,0,0),1,8);
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

	        for (int track = 0; track < tracked_humans.size(); track++) {
		        Scalar color(0,255,0);
		        TrackedObject human = tracked_humans[track];
		        circle(input_img, human.getPointBody(), 2, color, -1);
		        Point2f rect_points[4];
		        human.getBodyROI().points(rect_points);
		        for(int j = 0; j < 4; j++)
			        line( input_img, rect_points[j], rect_points[(j+1)%4], Scalar(192,192,0),1,8);
		        //circle(input_img, object.getPointBody(), 3*object.getSdBody(), Scalar(192,192,0), 2);

		        //circle(input_img, object.getPointHead(), 2, Scalar(192,192,192), -1);
		        human.getHeadROI().points(rect_points);
		        for(int j = 0; j < 4; j++)
			        line( input_img, rect_points[j], rect_points[(j+1)%4], Scalar(255,255,255),1,8);
		        //circle(input_img, object.getPointHead(), 3*object.getSdHead(), Scalar(255,0,0), 1);
		        int dir = human.getDirection();
		        float angle = (dir - human.getHeadROI().angle)*CV_PI/180.;
		        arrowedLine(input_img, human.getPointHead(), human.getPointHead() + 50.*Point2f(sin(angle), cos(angle)), Scalar(255,255,255), 1);
		        char buffer[10];
		        sprintf(buffer, "%d, %d", angles[track], human.getDirection());
		        putText(input_img, buffer , human.getBodyROI().center+50.*Point2f(-sin(human.getHeadROI().angle*CV_PI/180.),cos(human.getHeadROI().angle*CV_PI/180.)), FONT_HERSHEY_PLAIN, 1, Scalar(255,0,255), 2);
	        }

	        //imshow("FG Mask MOG 2", fgMaskMOG2);
	        imshow("Detection", input_img);
        }
        if(save_video)
	        outputVideo << input_img;
        //waitKey(1);
        //std::cout << end-begin << std::endl;
        return (tracked_humans.size() > 0 || tracked_objects.size() > 0);
}

void BGSub::groupContours ( vector< vector<Point> > inputContours, vector<RotatedRect> &outputBoundingBoxes, vector<RotatedRect> &rawBoundingBoxes, double distanceThreshold ) {
    if (!inputContours.size())
        return;
    // inputContours should be sorted in descending area order
    outputBoundingBoxes.clear();
    rawBoundingBoxes.clear();
    //int j;
    for (vector< vector<Point> >::iterator it = inputContours.begin(); it != inputContours.end(); ++it) {
        if (contourArea(*it) < area_threshold)          // Too small to be the seed
            break;
        vector<Point> contour_i = *it;
        RotatedRect rect_i = minAreaRect(contour_i);
        Point2f center_i = rect_i.center;
        double r_i = max(rect_i.size.width, rect_i.size.height) /2.;
        vector< vector<Point> >::iterator it_j = it+1;
        while (it_j != inputContours.end()) {
            vector<Point> contour_j = *it_j;
            RotatedRect rect_j = minAreaRect(contour_j);
            Point2f center_j = rect_j.center;
            double r_j = max(rect_j.size.width, rect_j.size.height) /2.;
            double d_ij = norm(center_i - center_j);        // Distance between 2 contours
            if ((d_ij - r_i - r_j) < distanceThreshold * (r_i+r_j)) {
                // Close - should be combined
                //cout << "\tMerged: " << it-inputContours.begin() << " and " << it_j-inputContours.begin() << endl;
                contour_i.insert(contour_i.end(), contour_j.begin(), contour_j.end());
                // update bounding box
                rect_i = minAreaRect(contour_i);
                r_i = max(rect_i.size.width, rect_i.size.height) /2.;
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

        Size human_size = getHumanSize(norm(center - img_center)) + Size(10,20);
        outputBoundingBoxes.push_back(RotatedRect(center, Size(max(int(cvRound(w_aligned)),human_size.width), max(int(cvRound(h_aligned)),human_size.height)), theta));
    }
}

RotatedRect BGSub::groupBlimp ( vector< vector<Point> > &inputContours, double distanceThreshold ) {
    if (!inputContours.size())
        return RotatedRect(Point2f(0.,0.), Size2f(0.,0.),0.);
    // inputContours should be sorted in descending area order
    vector< vector<Point> >::iterator it = inputContours.begin();
    vector<Point> contour_i = *it;
    if (contourArea(contour_i) < 100.)
        return RotatedRect(Point2f(0.,0.), Size2f(0.,0.),0.);
    RotatedRect rect_i = minAreaRect(contour_i);
    Point2f center_i = rect_i.center;
    double r_i = max(rect_i.size.width, rect_i.size.height) /2.;
    vector< vector<Point> >::iterator it_j = it+1;
    while (it_j != inputContours.end()) {
        vector<Point> contour_j = *it_j;
        RotatedRect rect_j = minAreaRect(contour_j);
        Point2f center_j = rect_j.center;
        double r_j = max(rect_j.size.width, rect_j.size.height) /2.;
        double d_ij = norm(center_i - center_j);        // Distance between 2 contours
        if ((d_ij - r_i - r_j) < distanceThreshold * (r_i+r_j)) {
            // Close - should be combined
            contour_i.insert(contour_i.end(), contour_j.begin(), contour_j.end());
            // update bounding box
            rect_i = minAreaRect(contour_i);
            r_i = max(rect_i.size.width, rect_i.size.height) /2.;
            it_j = inputContours.erase(it_j);
        }
        else {
            ++it_j;
        }
    }
    return rect_i;
}


Size BGSub::getHumanSize(float radius) {
	float width;
	if (radius > 280)
		width = 24.;
	else if (radius < 120)
		width = 88.;
	else
		width = cvRound(136.26 - 0.4*radius);
	return Size(width, 2*width);
}
