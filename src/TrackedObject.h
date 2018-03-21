#ifndef TrackedObject_h
#define TrackedObject_h

#include <opencv/highgui.h>
#include <cvaux.h>
#include <math.h>
#include <opencv2/opencv.hpp>

using namespace cv;

//////// DetectedObjects ////////
enum objectStatus {
	OBJ = 0,
	HUMAN = 1
};

class TrackedObject {
public:
	TrackedObject(RotatedRect objDetection, bool isHumanDetected, bool isHeadDetected, RotatedRect headDetection, Point2f imgCenter);
	Point2f PredictObject();
	Point2f UpdateObject(RotatedRect objDetection, bool isHumanDetected);			// Return predicted position of the head area
	//Point2f PredictHead(Mat &obj_vel);
	Point2f UpdateHead(RotatedRect headDetection);
	// For checking if the new detection belongs to this objects
	bool IsForThisObject(RotatedRect new_obj);
	// For checking if the new detection belongs to this objects; return distance to this object
	float distToObject(RotatedRect new_obj);
	// For checking if the new head belongs to this objects
	bool IsForThisHead(RotatedRect new_head);
	// For checking if the new head belongs to this objects; return distance to this object's head prediction
	float distToHead(RotatedRect new_head);
	// Final check at the end of the loop: if variance becomes too large, remove
	// Return true if deleted
	bool CheckAndDelete();
	float getSdBody();
	float getSdHead();
	Point2f getPointBody();
	Point2f getPointHead();
	RotatedRect getBodyROI();
	RotatedRect getHeadROI();
	Point2f getHeadVel();
	int getStatus();
	float threshold();
	int getDirection();
	void updateDirection(int estimation, int movingDirection);
	int getCount();

private:
	KalmanFilter objectKF;
	KalmanFilter headKF;
	RotatedRect objectROI;
	RotatedRect headROI;
	int status;
	int countHuman;
	float sdBody;
	float sdHead;

	// To keep body and head widths (constanted, changed only by new detections
	float bodyWidth;
	float headWidth;

	// For estimating head position & size based on human detection
	float heightRatio;		// head's height relative to window's height
	float deltaAngle;
	float headRatio;

	// For tracking of head direction
	int headDirection;

	Point2f img_center;
};

#endif
