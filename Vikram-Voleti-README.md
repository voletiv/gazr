## Find the head pose in multiple frame files.

- Compile

```sh
mkdir build
cd build
cmake ..
make
```

- Run (from inside build)

```sh
./benchmark_head_pose_estimation_multiple_frames ../../lipreading-in-the-wild-experiments/shape-predictor/shape_predictor_68_face_landmarks.dat ../tools/imageFileNames.txt > head_poses.txt
```
  - Input 1: full path to shape_predictor_68_face_landmarks
  - Input 2: imageFileNames.txt - list of frame file names "/home/voletiv/Datasets/GRIDcorpus/s01/bbbaf2n/bbaf2nFrame01.jpg\n/home/voletiv/Datasets/GRIDcorpus/s01/bbbaf2n/bbaf2nFrame02.jpg..." 
  - Save all head poses in head_poses.txt: ```> head_poses.txt```
