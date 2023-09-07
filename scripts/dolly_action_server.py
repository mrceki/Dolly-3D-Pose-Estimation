#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import LaserScan
from actionlib import SimpleActionServer

from dolly_pose_estimation.msg import DollyPoseAction, DollyPoseGoal
from dolly_pose_estimation.msg import DollyPoseFeedback, DollyPoseResult

import dolly_utils as utils

class DollyPoseEstimationServer:
    def __init__(self):
        self.server = SimpleActionServer('dolly_pose_estimation_server',  DollyPoseAction, self.execute, False)
        self.is_processing = False
        self.server.start()

    def execute(self, goal: DollyPoseGoal):
        DOLLY_SIZE_X = rospy.get_param("/dolly_pose_estimation_server/dolly_size_x", 1.12895)
        DOLLY_SIZE_Y = rospy.get_param("/dolly_pose_estimation_server/dolly_size_y", 1.47598)
        dolly_dimensions = [DOLLY_SIZE_X, DOLLY_SIZE_Y]
        scan_range = rospy.get_param("/dolly_pose_estimation_server/scan_range", 3.0)
        dbscan_eps = rospy.get_param("/dolly_pose_estimation_server/dbscan_eps", 0.1)
        dbscan_min_samples = rospy.get_param("/dolly_pose_estimation_server/dbscan_min_samples", 1)
        dbscan_max_samples = rospy.get_param("/dolly_pose_estimation_server/dbscan_max_samples", 6)
        dolly_dimension_tolerance = rospy.get_param("/dolly_pose_estimation_server/dolly_dimension_tolerance", 0.15)
        scan_topic = rospy.get_param("/dolly_pose_estimation_server/scan_topic", "/scan")
        feedback = DollyPoseFeedback()
        result = DollyPoseResult()
        if goal.cancel:
            self.is_processing = False
            result =  DollyPoseResult()
            result.success = False
            # result.message = "Dolly pose estimation canceled."
            self.server.set_preempted(result)
            return

        self.is_processing = True
        while self.is_processing:
            scan_data = rospy.wait_for_message(scan_topic, LaserScan, timeout=5)
            cartesian_points = utils.cartesian_conversion(scan_data, scan_range)

            if cartesian_points == []:
                feedback.message = "Can't find any laser"
                feedback.success = False
                self.server.publish_feedback(feedback)
                result.success = False
                continue
            
            clusters = utils.dbscan_clustering(cartesian_points, dbscan_max_samples, dolly_dimension_tolerance, dbscan_eps, dbscan_min_samples, scan_range, dolly_dimensions)
            num_clusters = len(clusters)
            utils.dolly_check(num_clusters)

            if num_clusters < 4:
                feedback.message = "Not enough clusters found."
                feedback.success = False
                self.server.publish_feedback(feedback)
                continue

            kmeans, sorted_clusters = utils.kmeans_clustering(clusters, num_clusters // 4)
            dolly_poses = utils.calculate_dolly_poses(kmeans, sorted_clusters, dolly_dimension_tolerance, dolly_dimensions)       
            utils.publish_transforms(dolly_poses, sorted_clusters)
            respond = utils.generate_poseArray(dolly_poses)
            
            if self.server.is_preempt_requested():
                self.server.set_preempted()
                self.is_processing = False
                return

            feedback.message = f"{int(num_clusters/4)} dolly found."
            feedback.poses = respond
            feedback.success = True 
            self.server.publish_feedback(feedback)
            # self.server.set_preempted(result)

def main():
    rospy.init_node('dolly_pose_estimation_server')
    dolly_server = DollyPoseEstimationServer()
    rospy.spin()

if __name__ == '__main__':
    main()
