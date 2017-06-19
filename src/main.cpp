#include <string>
#include <ros/ros.h>

#include "ros_head_pose_estimator.hpp"
#include "facialfeaturescloud.hpp"

using namespace std;

int main(int argc, char* argv[])
{
    //ROS initialization
    ros::init(argc, argv, "ros_faces");
    ros::NodeHandle rosNode;
    ros::NodeHandle _private_node("~");

    // load parameters
    string modelFilename;
    _private_node.param<string>("face_model", modelFilename, "");

    string prefix;
    _private_node.param<string>("prefix", prefix, "face");

    bool enableDepth;
    _private_node.param<bool>("with_depth", enableDepth, false);

    if (modelFilename.empty()) {
        ROS_ERROR_STREAM("You must provide the face model with the parameter face_model.\n" <<
                         "For instance, _face_model:=shape_predictor_68_face_landmarks.dat");
        return(1);
    }

    // initialize the detector by subscribing to the camera video stream
    ROS_INFO_STREAM("Initializing the face detector with the model " << modelFilename <<"...");
    if(!enableDepth) {
        HeadPoseEstimator estimator(rosNode, prefix, modelFilename);
        ROS_INFO_STREAM("RGB-only estimator successfully initialized." << endl <<
                        "TF frames of detected faces will be published when detected," << endl <<
                        "as well as the nb of detected faces on /nb_detected_faces.");
        ros::spin();
    }
    else {
        FacialFeaturesPointCloudPublisher estimator(rosNode, prefix, modelFilename);
        ROS_INFO_STREAM("RGB-D estimator successfully initialized." << endl <<
                        "TF frames of detected faces will be published when detected," << endl <<
                        "point clouds of 3D facial features will be made available on /facial_features," << endl <<
                        "the nb of detected faces will be published on /nb_detected_faces.");
        ros::spin();
    }


    return 0;
}

