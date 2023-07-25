#!/usr/bin/env python3

import rospy
from dolly_pose_estimation.srv import DollyPose
from sensor_msgs.msg import LaserScan

def dolly_pose_estimation_client():
    rospy.wait_for_service('/dolly_pose_estimation')
    try:
        dolly_pose_estimation = rospy.ServiceProxy('/dolly_pose_estimation', DollyPose)

        scan_data = rospy.wait_for_message('/diffbot/scan', LaserScan)

        response = dolly_pose_estimation(scan_data)
        dolly_positions = response.poses.poses
        
        for i, pose in enumerate(dolly_positions):
            print(f"Dolly {i}: X = {pose.position.x}, Y = {pose.position.y}, Yaw = {pose.orientation.z}")

    except rospy.ServiceException as e:
        print("Service call failed:", str(e))

if __name__ == '__main__':
    rospy.init_node('dolly_pose_estimation_client')
    dolly_pose_estimation_client()
