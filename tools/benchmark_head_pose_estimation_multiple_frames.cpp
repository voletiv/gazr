#define STR_EXPAND(tok) #tok
#define STR(tok) STR_EXPAND(tok)

#ifdef OPENCV3
#include <opencv2/imgcodecs.hpp>
#else
#include <opencv2/highgui/highgui.hpp>
#endif

#include <iostream>

#include "../src/head_pose_estimation.hpp"

using namespace std;
using namespace cv;

const static size_t NB_TESTS = 2; // number of time the detection is run, to get better average detection duration

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

int main(int argc, char **argv)
{
    Mat frame;

    if(argc < 3) {
        cerr << argv[0] << " " << STR(GAZR_VERSION) << "\n\nUsage: " 
             << endl << argv[0] << " model.dat fileNames.txt" << endl;
        return 1;
    }


    auto estimator = HeadPoseEstimation(argv[1]);
    estimator.focalLength = 500;

    // Read file
    std::string frameFilesTxt(argv[2]);
    std::vector<std::string> frameFileNames = readFileToVector(frameFilesTxt);

    cout << "Running " << NB_TESTS << " loops to get a good performance estimate..." << endl;

    auto t_start = getTickCount();

    for(auto frameFileName: frameFileNames) {

        cout << "Estimating head pose on " << frameFileName << endl;
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

        // cout << "Found " << nbfaces/NB_TESTS << " faces" << endl;

        auto poses = estimator.poses();
        for(auto pose : poses) {
        cout << "Head pose: (" << pose(0,3) << ", " << pose(1,3) << ", " << pose(2,3) << ")" << endl;

        }

    }

    auto t_end = getTickCount();
    // cout << "Face feature detection: " <<((t_detection-t_start) / NB_TESTS) /getTickFrequency() * 1000. << "ms;";
    // cout << "Pose estimation: " <<((t_end-t_detection) / NB_TESTS) /getTickFrequency() * 1000. << "ms;";
    cout << "Total time: " << (t_end-t_start) / getTickFrequency() * 1000. << "ms" << endl;

}
