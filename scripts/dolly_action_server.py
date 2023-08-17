#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import LaserScan
from actionlib import SimpleActionServer

from dolly_pose_estimation.msg import DollyPoseAction, DollyPoseGoal
from dolly_pose_estimation.msg import DollyPoseFeedback, DollyPoseResult

import dolly_utils as utils

# Size of dolly
DOLLY_SIZE_X = 1.12895
DOLLY_SIZE_Y = 1.47598
DOLLY_SIZE_HYPOTENUSE = (DOLLY_SIZE_X ** 2 + DOLLY_SIZE_Y ** 2) ** 0.5


class DollyPoseEstimationServer:
    def __init__(self):
        self.server = SimpleActionServer('dolly_pose_estimation',  DollyPoseAction, self.execute, False)
        self.is_processing = False
        self.server.start()

    def execute(self, goal: DollyPoseGoal):
        feedback = DollyPoseFeedback()
        if goal.cancel:
            self.is_processing = False
            result =  DollyPoseResult()
            result.success = False
            result.message = "Dolly pose estimation canceled."
            self.server.set_preempted(result)
            return

        self.is_processing = True
        while self.is_processing:
            scan_data = rospy.wait_for_message("/scan", LaserScan, timeout=5)
            cartesian_points = utils.cartesian_conversion(scan_data)

            if cartesian_points == []:
                feedback.message = "Can't find any laser"
                self.server.publish_feedback(feedback)
                continue
            
            clusters = utils.dbscan_clustering(cartesian_points)
            num_clusters = len(clusters)
            utils.dolly_check(num_clusters)

            if num_clusters < 4:
                feedback.message = "Not enough clusters found."
                self.server.publish_feedback(feedback)
                continue

            kmeans, sorted_clusters = utils.kmeans_clustering(clusters, num_clusters // 4)
            dolly_poses = utils.calculate_dolly_poses(kmeans, sorted_clusters)       
            utils.publish_transforms(dolly_poses, sorted_clusters)
            respond = utils.generate_poseArray(dolly_poses)
            
            if self.server.is_preempt_requested():
                self.server.set_preempted()
                self.is_processing = False
                return

            feedback.message = f"{int(num_clusters/4)} dolly found."
            feedback.poses = respond
            self.server.publish_feedback(feedback)
                 
        result = DollyPoseResult()
        self.server.set_succeeded(result)

def main():
    rospy.init_node('dolly_pose_estimation_server')
    dolly_server = DollyPoseEstimationServer()
    rospy.spin()

if __name__ == '__main__':
    main()
