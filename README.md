# Dolly Pose Estimation

This repository contains Python code for estimating the pose of dolly objects using laser scan data. The code performs clustering, sorting, and pose estimation for dolly objects within a laser scan's field of view.

## Dependencies

Before running the code, make sure you have the following dependencies installed:

- ROS (Robot Operating System)
- Python 3
- `sklearn` (Scikit-Learn)
- `numpy`
- `tf` (Transform Library for ROS)

### You can install the necessary ROS dependencies using `rosdep` with the following command:

```
rosdep install --from-paths /path/to/your/catkin_ws/src --ignore-src -r -y
```
### Running the Code
To run the code, follow these steps:

- Make sure you have ROS installed and configured properly.

- Clone this repository into your ROS workspace's source directory.

- Build your ROS workspace:

```
cd /path/to/your/catkin_ws
catkin_make
```
- Source your ROS workspace:
```
source /path/to/your/catkin_ws/devel/setup.bash
```
- Launch the pose estimation node using the provided script:
```
rosrun dolly_pose_estimation dolly.py
```
- Launch the pose estimation action server using the provided launch file:
```
roslaunch dolly_pose_estimation dolly_pose_estimation.launch
```
- Send an action goal request to start pose estimation
```
rostopic pub /dolly_pose_estimation_server/goal dolly_pose_estimation/DollyPoseActionGoal {} --once
```
- Send an action goal request to stop pose estimation
```
rostopic pub /dolly_pose_estimation_server/cancel actionlib_msgs/GoalID {} --once
```
The node will start processing laser scan data and estimating dolly poses. You can also use the provided action interface to request pose estimation.

Launch File Parameters
The launch file `dolly_pose_estimation.launch` accepts several parameters that you can customize:

- `dolly_size_x` and `dolly_size_y`: Dimensions of the dolly in the X and Y directions, respectively.
- `dolly_dimension_tolerance`: Tolerance for dolly dimension comparison during sorting.
- `scan_range`: Maximum range for laser scan data.
- `dbscan_eps`: DBSCAN clustering parameter (maximum distance between points in a cluster).
- `dbscan_min_samples` and `dbscan_max_samples`: Minimum and maximum number of samples for DBSCAN clustering.
- `scan_topic`: Laser scan topic to subscribe to.
