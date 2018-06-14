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
	setIdentity(objectKF.processNoiseCov, Scalar::all(25.0));
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
	setIdentity(headKF.processNoiseCov, Scalar::all(25.0));
	setIdentity(headKF.measurementNoiseCov, Scalar::all(sdHead*sdHead));
	setIdentity(headKF.errorCovPost, Scalar::all(sdHead*sdHead));
	if (isHumanDetected)
		countHuman = 1;
	else
		countHuman = 0;
	if (isHeadDetected) {
	    Point2f bodyToHead = headDetection.center - objDetection.center;
		heightRatio = norm(bodyToHead)/ objDetection.size.height;
		deltaAngle = atan2(bodyToHead.x, -bodyToHead.y) - objDetection.angle;
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
		float headAngle = atan2(headCenter.x, -headCenter.y);
		headROI = RotatedRect(headCenter, Size(headWidth, headWidth), headAngle);
	}
	//setIdentity(headKF.transitionMatrix);
	//setIdentity(headKF.controlMatrix);
	//setIdentity(headKF.measurementMatrix);
	//setIdentity(headKF.measurementNoiseCov, Scalar::all(sdHead*sdHead));
	//setIdentity(headKF.errorCovPost, Scalar::all(sdHead*sdHead));
	status = OBJ;

	headDirection = -1;			// Init
    currentEstimation = -1;
    currentMovingDirection = -1;

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
        if (countHuman == 1)        // the first time detected as human, should not keep previous size
            bodyWidth = objDetection.size.width;
        else                        // average with the old one to smooth the width
            bodyWidth = 0.5*(bodyWidth + objDetection.size.width);
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

Point2f TrackedObject::updateHeadfromBody() {
    // After updating the object, use it as weak measurement of head
    float theta_r = (objectROI.angle + deltaAngle)*CV_PI/180.;
    Point2f headCenter = objectROI.center + heightRatio*objectROI.size.height*Point2f(sin(theta_r), -cos(theta_r));
    headWidth = headRatio*objectROI.size.width;

    if (countNonZero(headKF.statePre) == 0) {             // Just created, without any prediction performed
        headKF.statePre = (Mat_<float>(4,1) << headCenter.x, headCenter.y, 0, 0);
        headROI = RotatedRect(headCenter, Size(headWidth, headWidth),
                              atan2(headCenter.x - img_center.x, img_center.y - headCenter.y) *180./CV_PI);
        return headROI.center;
    }

    Mat measurement_frombody = (Mat_<float>(2,1) << headCenter.x, headCenter.y);
    if (headWidth < 6.) {
        // Too small and should be limited to prevent problems
        headWidth = 6.;
        headRatio = headWidth/objectROI.size.width;
    }
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

    return headROI.center;
}

Point2f TrackedObject::UpdateHead(RotatedRect headDetection) {
    Mat measurement = (Mat_<float>(2,1) << headDetection.center.x, headDetection.center.y);
    headWidth = headDetection.size.width;
    setIdentity(headKF.measurementNoiseCov, Scalar::all(headDetection.size.width*headDetection.size.width/16.));
    Mat corrected_state = headKF.correct(measurement);
    sdHead = sqrt(min(headKF.errorCovPost.at<float>(0,0), headKF.errorCovPost.at<float>(1,1)));
    headROI = RotatedRect(Point2f(corrected_state.at<float>(0,0), corrected_state.at<float>(1,0)),
                          Size2f(headWidth, headWidth),
                          atan2(corrected_state.at<float>(0,0) - img_center.x, img_center.y - corrected_state.at<float>(1,0)) *180./CV_PI);
    Point2f bodyToHead = headROI.center - objectROI.center;
    heightRatio = norm(bodyToHead) / objectROI.size.height;
    deltaAngle = atan2(bodyToHead.x, -bodyToHead.y)*180./CV_PI - objectROI.angle;
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
    currentEstimation = estimation;
    currentMovingDirection = movingDirection;
    if (headDirection < 0) {
        if (movingDirection >= 0) {
            if (estimation >= 0) {
                if (abs(movingDirection - estimation) >= 180)
                    headDirection = cvRound((movingDirection + estimation + 360)/2.);
                else
                    headDirection = cvRound((movingDirection + estimation)/2.);
            }
            else {
                headDirection = movingDirection;
            }
        }
        else {
            headDirection = estimation;
        }
        return;
    }

    if (estimation < 0) {
        // When the head is out of the frame and direction cannot be estimated
        // Use only moving direction
        if (movingDirection >= 0) {
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
    if (diff <= 45) {               // <= 45 degree change
        headDirection = cvRound((headDirection + estimation)/2.);
    }
    else if ( diff >= 315) {        // <= 45 degree change, crossing the line 0,360
        headDirection = cvRound((headDirection + estimation + 360)/2.);
        if (headDirection >= 360)
            headDirection -= 360;
    }
    // More than that, update with the moving direction instead
    else {
        if (movingDirection > 0) {
            // Moving
            if (abs(movingDirection - headDirection) >= 180) {
                // crossing 0,360 line
                headDirection = cvRound((headDirection + movingDirection + 360)/2.);
                if (headDirection >= 360)
                    headDirection -= 360;
            }
            else {
                headDirection = cvRound((headDirection + movingDirection)/2.);
            }
        }
    }
}

string TrackedObject::getStringForSave() {
    std::ostringstream ss;
    ss << countHuman << ",";
    ss << objectROI.center.x << "," << objectROI.center.y << "," << objectROI.size.width << "," << objectROI.size.height << "," << objectROI.angle << ",";
    ss << headROI.center.x << "," << headROI.center.y << "," << headROI.size.width << "," << headROI.size.height << "," << headROI.angle << ",";
    ss << heightRatio << "," << headRatio << "," << deltaAngle << ",";
    ss << currentEstimation << "," << currentMovingDirection << "," << headDirection;
    return ss.str();
}

bool TrackedObject::isTrackedHeadInvalid() {
    return (heightRatio > 0.5 || heightRatio < 0.1 || headRatio > 0.8 || headRatio < 0.1 || fabs(deltaAngle) > 26.6);       // atan(0.5)
}

bool TrackedObject::isNewHeadLegit(RotatedRect head) {
    float heightR = norm(head.center - objectROI.center)/objectROI.size.height;
    float headR = head.size.width/bodyWidth;
    float deltaAng = head.angle - objectROI.angle;
    while (deltaAng > 180.)
        deltaAng -= 360.;
    return (heightR <= 0.5 && heightR >= 0.1 && headR <= 0.8 && headR >= 0.1 && fabs(deltaAng) <= 26.6);
}
/////////////////////////////////
