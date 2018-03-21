#include <opencv2/opencv.hpp>
#include <math.h>
#include "TrackedObject.h"

using namespace cv;

TrackedObject::TrackedObject(RotatedRect objDetection, bool isHumanDetected, bool isHeadDetected, RotatedRect headDetection, Point2f imgCenter) {
	objectKF = KalmanFilter(4, 2, 0);
	//headKF = KalmanFilter(3, 3, 3);
	headKF = KalmanFilter(4, 2, 0);
	sdBody = objDetection.size.width/4.;
	sdHead = objDetection.size.width/4.;

	objectKF.transitionMatrix = (Mat_<float>(4,4) << 1,0,1,0,
													 0,1,0,1,
													 0,0,1,0,
													 0,0,0,1);
	setIdentity(objectKF.measurementMatrix);
	setIdentity(objectKF.processNoiseCov, Scalar::all(16.0));
	setIdentity(objectKF.measurementNoiseCov, Scalar::all(sdBody*sdBody));
	setIdentity(objectKF.errorCovPost, Scalar::all(sdBody*sdBody));
	objectKF.statePost = (Mat_<float>(4,1) << objDetection.center.x, objDetection.center.y, 0, 0);
	bodyWidth = objDetection.size.width;
	objectROI = objDetection;

	headKF.transitionMatrix = (Mat_<float>(4,4) << 1,0,1,0,
												   0,1,0,1,
												   0,0,1,0,
												   0,0,0,1);
	setIdentity(headKF.measurementMatrix);
	setIdentity(headKF.processNoiseCov, Scalar::all(16.0));
	setIdentity(headKF.measurementNoiseCov, Scalar::all(sdHead*sdHead));
	setIdentity(headKF.errorCovPost, Scalar::all(sdHead*sdHead));
	if (isHumanDetected)
		countHuman = 1;
	else
		countHuman = 0;
	if (isHeadDetected) {
		heightRatio = norm(headDetection.center - objDetection.center)/ objDetection.size.height;
		deltaAngle = headDetection.angle - objDetection.angle;
		headRatio = headDetection.size.width/objDetection.size.width;
		headKF.statePost = (Mat_<float>(4,1) << headDetection.center.x, headDetection.center.y, 0, 0);
		headWidth = headDetection.size.width;
		headROI = headDetection;
	}
	else {
		heightRatio = 0.3125;
		deltaAngle = 0.;
		headRatio = 0.375;
		float theta_r = (objDetection.angle + deltaAngle)*CV_PI/180.;
		Point2f headCenter = objDetection.center + heightRatio*objDetection.size.height*Point2f(sin(theta_r), -cos(theta_r));
		headKF.statePost = (Mat_<float>(4,1) << headCenter.x, headCenter.y, 0, 0);
		headWidth = headRatio*objDetection.size.width;
		headROI = RotatedRect(headCenter, Size(headWidth, headWidth), objDetection.angle);
	}
	//setIdentity(headKF.transitionMatrix);
	//setIdentity(headKF.controlMatrix);
	//setIdentity(headKF.measurementMatrix);
	//setIdentity(headKF.measurementNoiseCov, Scalar::all(sdHead*sdHead));
	//setIdentity(headKF.errorCovPost, Scalar::all(sdHead*sdHead));
	status = OBJ;

	headDirection = -1;			// Init

	img_center = imgCenter;
}

Point2f TrackedObject::PredictObject() {
	Mat prediction = objectKF.predict();
	sdBody = sqrt(min(objectKF.errorCovPost.at<float>(0,0), objectKF.errorCovPost.at<float>(1,1)));
	objectROI = RotatedRect(Point2f(prediction.at<float>(0,0), prediction.at<float>(1,0)),
							Size2f(bodyWidth, 2*bodyWidth),
							atan2(prediction.at<float>(0,0) - img_center.x, img_center.y - prediction.at<float>(1,0)) *180./CV_PI);

	Mat predictHead = headKF.predict();
	sdHead = sqrt(min(headKF.errorCovPost.at<float>(0,0), headKF.errorCovPost.at<float>(1,1)));
	headROI = RotatedRect(Point2f(predictHead.at<float>(0,0), predictHead.at<float>(1,0)),
						  Size2f(headWidth, headWidth),
						  atan2(predictHead.at<float>(0,0) - img_center.x, img_center.y - predictHead.at<float>(1,0)) *180./CV_PI);

	return Point2f(prediction.at<float>(0,0), prediction.at<float>(1,0));

	/*float r = norm(getPointBody()-img_center);
	float l = norm(getPointHead()-img_center);
	float w1 = objectROI.size.width;
	Point2f unit_r1 = (1./r) * (getPointBody()-img_center);
	Mat prediction = objectKF.predict();
	Point2f p2 = Point2f(prediction.at<float>(0,0), prediction.at<float>(1,0));
	float r2 = norm(p2-img_center);
	Point2f unit_r2 = (1./r2) * (p2-img_center);
	sdBody = sqrt(min(objectKF.errorCovPost.at<float>(0,0), objectKF.errorCovPost.at<float>(1,1)));
	objectROI = RotatedRect(p2,
							Size2f(prediction.at<float>(2,0), 2*prediction.at<float>(2,0)),
							atan2(prediction.at<float>(0,0) - img_center.x, img_center.y - prediction.at<float>(1,0)) *180./CV_PI);
	headKF.processNoiseCov = objectKF.errorCovPost(Rect(3,3,3,3));
	Point2f vel1(objectKF.statePost.at<float>(3,0), objectKF.statePost.at<float>(4,0));
	Point2f vel2 = vel1 + (l-r)*(unit_r2-unit_r1) - 0.4*(r2-r)*(l-r)/w1*unit_r2;

	//cout << vel1 << "\t" << vel2;
	Mat obj_vel = (Mat_<float>(3,1) << vel2.x, vel2.y, 0.4*prediction.at<float>(5,0));

	Mat predictHead = headKF.predict(obj_vel);
	sdHead = sqrt(min(headKF.errorCovPost.at<float>(0,0), headKF.errorCovPost.at<float>(1,1)));
	headROI = RotatedRect(Point2f(predictHead.at<float>(0,0), predictHead.at<float>(1,0)),
						  Size2f(predictHead.at<float>(2,0), predictHead.at<float>(2,0)),
						  atan2(predictHead.at<float>(0,0) - img_center.x, img_center.y - predictHead.at<float>(1,0)) *180./CV_PI);
	return Point2f(prediction.at<float>(0,0), prediction.at<float>(1,0));*/
}

Point2f TrackedObject::UpdateObject(RotatedRect objDetection, bool isHumanDetected) {
	Mat measurement;
	if (isHumanDetected) {
		if (status == OBJ) {
			countHuman++;
			if (countHuman >= 3)
				status = HUMAN;
		}
		setIdentity(objectKF.measurementNoiseCov, Scalar::all(objDetection.size.width*objDetection.size.width/16.));
		measurement = (Mat_<float>(2,1) << objDetection.center.x, objDetection.center.y);
		bodyWidth = objDetection.size.width;
	}
	else {
		setIdentity(objectKF.measurementNoiseCov, Scalar::all(objDetection.size.width*objDetection.size.width/16.));		// Larger variance for object
		measurement = (Mat_<float>(2,1) << objDetection.center.x, objDetection.center.y);
		bodyWidth = objectROI.size.width;
	}
	Mat corrected_state = objectKF.correct(measurement);
	objectROI = RotatedRect(Point2f(corrected_state.at<float>(0,0), corrected_state.at<float>(1,0)),
							Size2f(bodyWidth, 2*bodyWidth),
							atan2(corrected_state.at<float>(0,0) - img_center.x, img_center.y - corrected_state.at<float>(1,0)) *180./CV_PI);

	// After updating the object, use it as weak measurement of head
	float theta_r = (objectROI.angle + deltaAngle)*CV_PI/180.;
	Point2f headCenter = objectROI.center + heightRatio*objectROI.size.height*Point2f(sin(theta_r), -cos(theta_r));
	Mat measurement_frombody = (Mat_<float>(2,1) << headCenter.x, headCenter.y);
	headWidth = headRatio*objectROI.size.width;
	setIdentity(headKF.measurementNoiseCov, Scalar::all(objectROI.size.width*objectROI.size.width/16.));
	Mat corrected_head = headKF.correct(measurement_frombody);
	headROI = RotatedRect(Point2f(corrected_head.at<float>(0,0), corrected_head.at<float>(1,0)),
						  Size(headWidth, headWidth),
						  atan2(corrected_head.at<float>(0,0) - img_center.x, img_center.y - corrected_head.at<float>(1,0)) *180./CV_PI);
	//heightRatio = 0.3125;
	//deltaAngle = 0.;
	//headRatio = 0.375;

	//Mat obj_vel = corrected_state.rowRange(3,6);
	sdBody = sqrt(min(objectKF.errorCovPost.at<float>(0,0), objectKF.errorCovPost.at<float>(1,1)));
	sdHead = sqrt(min(headKF.errorCovPost.at<float>(0,0), headKF.errorCovPost.at<float>(1,1)));
	return Point2f(corrected_state.at<float>(0,0), corrected_state.at<float>(1,0));
}

/*Point2f TrackedObject::PredictHead(Mat &obj_vel) {
	Mat prediction = headKF.predict(obj_vel);
	sdHead = sqrt(min(headKF.errorCovPost.at<float>(0,0), headKF.errorCovPost.at<float>(1,1)));
	headROI = RotatedRect(Point2f(prediction.at<float>(0,0), prediction.at<float>(1,0)),
						  Size2f(prediction.at<float>(2,0), prediction.at<float>(2,0)),
						  atan2(prediction.at<float>(0,0) - img_center.x, img_center.y - prediction.at<float>(1,0)) *180./CV_PI);
	return Point2f(prediction.at<float>(0,0), prediction.at<float>(1,0));
}*/

Point2f TrackedObject::UpdateHead(RotatedRect headDetection) {
	Mat measurement = (Mat_<float>(2,1) << headDetection.center.x, headDetection.center.y);
	headWidth = headDetection.size.width;
	setIdentity(headKF.measurementNoiseCov, Scalar::all(headDetection.size.width*headDetection.size.width/16.));
	Mat corrected_state = headKF.correct(measurement);
	sdHead = sqrt(min(headKF.errorCovPost.at<float>(0,0), headKF.errorCovPost.at<float>(1,1)));
	headROI = RotatedRect(Point2f(corrected_state.at<float>(0,0), corrected_state.at<float>(1,0)),
						  Size2f(headWidth, headWidth),
						  atan2(corrected_state.at<float>(0,0) - img_center.x, img_center.y - corrected_state.at<float>(1,0)) *180./CV_PI);
	heightRatio = norm(headROI.center - objectROI.center) / objectROI.size.height;
	deltaAngle = headROI.angle - objectROI.angle;
	headRatio = headROI.size.width/objectROI.size.width;
	return Point2f(corrected_state.at<float>(0,0), corrected_state.at<float>(1,0));
}

bool TrackedObject::IsForThisObject(RotatedRect new_obj) {
	Point2f predicted_pos(objectKF.statePost.at<float>(0,0), objectKF.statePost.at<float>(1,0));
	return (norm(new_obj.center-predicted_pos) < 3*sdBody);
}

float TrackedObject::distToObject(RotatedRect new_obj) {
	Point2f predicted_pos(objectKF.statePost.at<float>(0,0), objectKF.statePost.at<float>(1,0));
	return (norm(new_obj.center-predicted_pos));
}

bool TrackedObject::IsForThisHead(RotatedRect new_head) {
	Point2f predicted_pos(headKF.statePost.at<float>(0,0), headKF.statePost.at<float>(1,0));
	return (norm(new_head.center-predicted_pos) < 3*sdHead);
}

float TrackedObject::distToHead(RotatedRect new_head) {
	Point2f predicted_pos(headKF.statePost.at<float>(0,0), headKF.statePost.at<float>(1,0));
	return (norm(new_head.center-predicted_pos));
}

bool TrackedObject::CheckAndDelete() {
	return (sdBody > 1.5*objectROI.size.width || sdHead > 1.5*objectROI.size.width); // || (status != HUMAN && sdHead > 20));			// Deviation > 30
}

float TrackedObject::threshold() {
	float r = norm(getPointBody() - img_center);
	return max(min(136.26 - 0.4*r, 88.),12.);
}

float TrackedObject::getSdBody() {
	return sdBody;
}

float TrackedObject::getSdHead() {
	return sdHead;
}

Point2f TrackedObject::getPointBody() {
	return Point2f(objectKF.statePost.at<float>(0,0), objectKF.statePost.at<float>(1,0));
}

Point2f TrackedObject::getPointHead() {
	return Point2f(headKF.statePost.at<float>(0,0), headKF.statePost.at<float>(1,0));
}

RotatedRect TrackedObject::getBodyROI() {
	return objectROI;
}

RotatedRect TrackedObject::getHeadROI() {
	return headROI;
}

int TrackedObject::getStatus() {
	return status;
}

Point2f TrackedObject::getHeadVel() {
	return Point2f(headKF.statePost.at<float>(2,0), headKF.statePost.at<float>(3,0));
}

int TrackedObject::getDirection() {
	return headDirection;
}

int TrackedObject::getCount() {
	return countHuman;
}

void TrackedObject::updateDirection(int estimation, int movingDirection) {
	if (headDirection < 0) {
		headDirection = estimation;
		return;
	}

	if (estimation < 0) {
		// When the head is out of the frame and direction cannot be estimated
		// Use only moving direction
		if (movingDirection > 0) {
			// Moving
			if (abs(movingDirection - headDirection) >= 180) {
				// crossing 0,360 line
				headDirection = cvRound((headDirection + movingDirection + 360)/2.);
				if (headDirection > 360)
					headDirection -= 360;
			}
			else {
				headDirection = cvRound((headDirection + movingDirection)/2.);
			}
		}
		return;
	}

	int diff = abs(estimation - headDirection);
	if (diff <= 45) {				// <= 45 degree change
		headDirection = cvRound((headDirection + estimation)/2.);
	}
	else if ( diff >= 315) {		// <= 45 degree change, crossing the line 0,360
		headDirection = cvRound((headDirection + estimation + 360)/2.);
		if (headDirection > 360)
			headDirection -= 360;
	}
	// More than that, update with the moving direction instead
	else {
		if (movingDirection > 0) {
			// Moving
			if (abs(movingDirection - headDirection) >= 180) {
				// crossing 0,360 line
				headDirection = cvRound((headDirection + movingDirection + 360)/2.);
				if (headDirection > 360)
					headDirection -= 360;
			}
			else {
				headDirection = cvRound((headDirection + movingDirection)/2.);
			}
		}
	}
}
/////////////////////////////////
