cmake_minimum_required(VERSION 3.0.2)
project(dolly_pose_estimation)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  roscpp
  rospy
  sensor_msgs
  tf
  tf2_ros
  message_generation
  std_msgs
  actionlib
  actionlib_msgs
)

add_service_files(
  FILES
  DollyPose.srv
)

add_action_files(
  DIRECTORY action
  FILES DollyPose.action
)

generate_messages(
  DEPENDENCIES
  geometry_msgs sensor_msgs actionlib_msgs std_msgs actionlib
)

catkin_package(
  CATKIN_DEPENDS rospy message_runtime actionlib actionlib_msgs
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

add_executable(${PROJECT_NAME}_client src/dolly_client.cpp)
target_link_libraries(${PROJECT_NAME}_client ${catkin_LIBRARIES})
