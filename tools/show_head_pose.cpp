#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <iomanip>

#include "../src/head_pose_estimation.hpp"

using namespace std;
using namespace cv;

inline double todeg(double rad) {
    return rad * 180 / M_PI;
}

int main(int argc, char **argv)
{
    Mat frame;

    namedWindow("headpose");

    if(argc < 2) {
        cerr << "Usage: " << endl <<
                "head_pose model.dat" << endl;
        return 1;
    }


    auto estimator = HeadPoseEstimation(argv[1]);

    // Configure the video capture
    // ===========================

    VideoCapture video_in(0);

    // adjust for your webcam!
    video_in.set(CV_CAP_PROP_FRAME_WIDTH, 640);
    video_in.set(CV_CAP_PROP_FRAME_HEIGHT, 480);
    estimator.focalLength = 500;
    estimator.opticalCenterX = 320;
    estimator.opticalCenterY = 240;

    if(!video_in.isOpened()) {
        cerr << "Couldn't open camera" << endl;
        return 1;
    }


    while(true) {
        video_in >> frame;

        auto t_start = getTickCount();
        estimator.update(frame);

        for(auto pose : estimator.poses()) {

        cout << fixed << setprecision(4) << "Head pose: (" << pose(0,3) << ", " << pose(1,3) << ", " << pose(2,3) << ")" << endl;

        // compute the quaternion, taken from https://en.wikipedia.org/wiki/Rotation_formalisms_in_three_dimensions
        double qi = 0.5 * sqrt(1 + pose(0,0) - pose(1,1) - pose(2,2));
        double qj = 0.25 / qi * (pose(0,1) + pose(1,0));
        double qk = 0.25 / qi * (pose(0,2) + pose(2,0));
        double qr = 0.25 / qi * (pose(2,1) - pose(1,2));

        // compute yaw, pitch, roll
        double roll = atan2(2 * (qr * qi + qj * qk), 1 - 2 * (qi*qi + qj * qj));
        double pitch = asin(2 * (qr * qj - qk * qi));
        double yaw  = atan2(2 * (qr * qk + qi * qj), 1 - 2 * (qj * qj + qk * qk));

        cout << setprecision(1) << fixed << "Orientation: yaw=" << todeg(yaw) << ", pitch=" << todeg(pitch) << ", roll=" << todeg(roll) << endl;
        auto t_end = getTickCount();
        cout << "Processing time for this frame: " << (t_end-t_start) / getTickFrequency() * 1000. << "ms" << endl;

        }

#ifdef HEAD_POSE_ESTIMATION_DEBUG
        imshow("headpose", estimator._debug);
        if (waitKey(10) >= 0) break;
#endif

    }
}



