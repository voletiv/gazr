#define STR_EXPAND(tok) #tok
#define STR(tok) STR_EXPAND(tok)

#include <opencv2/opencv.hpp>

#ifdef OPENCV3
#include <opencv2/imgcodecs.hpp>
#else
#include <opencv2/highgui/highgui.hpp>
#endif

#include <boost/program_options.hpp>
#include <iostream>
#include <iomanip>

#include "LinearMath/Matrix3x3.h"
#include "../src/head_pose_estimation.hpp"

using namespace std;
using namespace cv;

const static size_t NB_TESTS = 1; // number of time the detection is run, to get better average detection duration

inline double todeg(double rad) {
    return rad * 180 / M_PI;
}

std::vector<std::string> readFileToVector(const std::string& filename)
{
    std::ifstream source;
    source.open(filename);
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(source, line))
    {
        lines.push_back(line);
    }
    return lines;
}

void calculate_yaw_pitch_roll(head_pose& pose)
{           
    pose = pose.inv();

    double raw_yaw, raw_pitch, raw_roll;
    tf::Matrix3x3 mrot(
            pose(0,0), pose(0,1), pose(0,2),
            pose(1,0), pose(1,1), pose(1,2),
            pose(2,0), pose(2,1), pose(2,2));
    mrot.getRPY(raw_roll, raw_pitch, raw_yaw);

    raw_roll = raw_roll - M_PI/2;
    raw_yaw = raw_yaw + M_PI/2;

    double yaw, pitch, roll;

    roll = raw_pitch;
    yaw = raw_yaw;
    pitch = -raw_roll;

    // cout << "\"face_" << i << "\":";
    cout << setprecision(4) << fixed << "{\"yaw\":" << todeg(yaw) << ", \"pitch\":" << todeg(pitch) << ", \"roll\":" << todeg(roll) << ",";
    cout << setprecision(4) << fixed << "\"x\":" << pose(0,3) << ", \"y\":" << pose(1,3) << ", \"z\":" << pose(2,3) << "},";

    cout << ", " << pose(0,3) << ", " << pose(1,3) << ", " << pose(2,3);
}


void estimate_head_pose_on_frameFileName(const std::string& frameFileName, HeadPoseEstimation estimator, std::vector<head_pose>& prev_poses, bool print_prev_poses)
{
    cout << frameFileName;
#ifdef OPENCV3
    Mat img = imread(frameFileName, IMREAD_COLOR);
#else
    Mat img = imread(frameFileName, CV_LOAD_IMAGE_COLOR);
#endif

    auto nbfaces = 0;

    for(size_t i = 0; i < NB_TESTS; i++) {
        estimator.update(img);
    }

    // auto t_detection = getTickCount();

    for(size_t i = 0; i < NB_TESTS; i++) {
        auto poses = estimator.poses();
        nbfaces += poses.size(); // this is completly artifical: the only purpose is to make sure the compiler does not optimize away estimator.poses()
    }

    // cout << "Found " << nbfaces/NB_TESTS << " face(s)" << endl;

    auto poses = estimator.poses();
    if (poses.size() > 0) {
        
        for(auto pose : poses) {
            calculate_yaw_pitch_roll(pose);
        }
        
        prev_poses = poses;
    
    }
    else if (print_prev_poses) {
        for(auto pose : prev_poses) {
            calculate_yaw_pitch_roll(pose);
        }
    }
    
    cout << endl;
}


int main(int argc, char **argv)
{
    Mat frame;

    if(argc < 3) {
        cerr << argv[0] << " " << STR(GAZR_VERSION) << "\n\nUsage: " 
             << endl << argv[0] << " model.dat frame.{jpg|png}\n\nOR\n\n"
             << endl << argv[0] << " model.dat filenames.txt" << endl;
#ifdef HEAD_POSE_ESTIMATION_DEBUG
        cerr <<  "Output: a new frame 'head_pose_<frame>.png'" << endl;
#endif
        return 1;
    }

    std::string fileName = argv[2];

    if (fileName.find(".jpg") == std::string::npos and
        fileName.find(".png") == std::string::npos and
        fileName.find(".txt") == std::string::npos) {
        cerr << "Please input a .jpg file, or .png file, or .txt file containing list of images in individual lines.";
        return 1;
    }

    auto estimator = HeadPoseEstimation(argv[1]);
    estimator.focalLength = 500;

    // cout << "Running " << NB_TESTS << " loops to get a good performance estimate..." << endl;

#ifdef HEAD_POSE_ESTIMATION_DEBUG
    cerr <<  "ATTENTION! The benchmark is compiled in DEBUG mode: the performance is no going to be good!!" << endl;
#endif

    // Prev pose (default)
    head_pose prev_pose = {
        -1,    -1,    -1,    -1.,
        -1,    -1,    -1,    -1.,
        -1,    -1,    -1,    -1.,
        0,     0,     0,     1
    };
    std::vector<head_pose> prev_poses;
    prev_poses.push_back(prev_pose);

    auto t_start = getTickCount();

    // Single image file
    if (fileName.find(".jpg") != std::string::npos or fileName.find(".png") != std::string::npos) {
        cout << ".jpg or .png file" << endl;
        estimate_head_pose_on_frameFileName(fileName, estimator, prev_poses, false);
    }

    // Multiple lines
    else {
        cout << ".txt file" << endl;
        // Read file
        std::string frameFilesTxt(fileName);
        std::vector<std::string> frameFileNames = readFileToVector(frameFilesTxt);

        if (frameFileNames.size() == 0) {
            cout << fileName << "does not exist, or has no image names" << endl;
        }

        for(auto frameFileName: frameFileNames) {
            if (frameFileName.find("jpg") != std::string::npos or frameFileName.find("png") != std::string::npos) {
                estimate_head_pose_on_frameFileName(frameFileName, estimator, prev_poses, false);
            }
        }
    }

    auto t_end = getTickCount();

    // cout << "Face feature detection: " <<((t_detection-t_start) / NB_TESTS) /getTickFrequency() * 1000. << "ms;";
    // cout << "Pose estimation: " <<((t_end-t_detection) / NB_TESTS) /getTickFrequency() * 1000. << "ms;";
    cout << "Total time: " << (t_end-t_start) / getTickFrequency() * 1000. << "ms" << endl;

#ifdef HEAD_POSE_ESTIMATION_DEBUG
    imwrite("head_pose.png", estimator._debug);
#endif

}
