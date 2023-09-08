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
        self.scan_topic = rospy.get_param("/dolly_pose_estimation_server/scan_topic", "/scan")
        self.scan_range = rospy.get_param("/dolly_pose_estimation_server/scan_range", 3.0)
        self.dolly_dimensions = [
            rospy.get_param("/dolly_pose_estimation_server/dolly_size_x", 1.12895),
            rospy.get_param("/dolly_pose_estimation_server/dolly_size_y", 1.47598)
        ]
        self.dbscan_eps = rospy.get_param("/dolly_pose_estimation_server/dbscan_eps", 0.1)
        self.dbscan_min_samples = rospy.get_param("/dolly_pose_estimation_server/dbscan_min_samples", 1)
        self.dbscan_max_samples = rospy.get_param("/dolly_pose_estimation_server/dbscan_max_samples", 6)
        self.dolly_dimension_tolerance = rospy.get_param("/dolly_pose_estimation_server/dolly_dimension_tolerance", 0.15)
        
        self.feedback = DollyPoseFeedback()
        self.result = DollyPoseResult()
        
        self.loop_rate_param = rospy.get_param("/dolly_pose_estimation_server/loop_rate", 1)
        if self.loop_rate_param > 0:
            self.loop_rate = rospy.Rate(self.loop_rate_param)
            rospy.loginfo(f"Loop rate is set to {self.loop_rate_param} Hz.")
        elif self.loop_rate_param < 0:            
            rospy.logwarn("Invalid loop rate parameter, using default value as 'nonstop'!!!!")
        else:
            rospy.loginfo("Loop rate is set to 'nonstop'.")
        
        self.server.start()

    def execute(self, goal: DollyPoseGoal):
        if goal.cancel:
            self.is_processing = False
            self.result.success = False
            self.result.status = "Dolly pose estimation canceled."
            self.server.set_preempted(self.result)
            return

        self.is_processing = True
        while self.is_processing and not rospy.is_shutdown():
            scan_data = rospy.wait_for_message(self.scan_topic, LaserScan, timeout=5)
            cartesian_points = utils.cartesian_conversion(scan_data, self.scan_range)

            if not cartesian_points:
                self.feedback.status = "Can't find any laser"
                self.feedback.success = False
                self.server.publish_feedback(self.feedback)
                self.result.success = False
                rospy.logwarn("There isn't any laser data!!!")
                if self.server.is_preempt_requested():
                    self.result.status = "There wasn't any laser data!!!"
                    self.server.set_preempted(self.result)
                    self.is_processing = False
                    return
                continue

            clusters = utils.dbscan_clustering(
                cartesian_points,
                self.dbscan_max_samples,
                self.dolly_dimension_tolerance,
                self.dbscan_eps,
                self.dbscan_min_samples,
                self.scan_range,
                self.dolly_dimensions
            )
            num_clusters = len(clusters)
            utils.dolly_check(num_clusters)

            if num_clusters < 4:
                self.feedback.status = "Not enough clusters found."
                self.feedback.success = False
                self.server.publish_feedback(self.feedback)
                if self.server.is_preempt_requested():
                    self.result.status = "There wasn't enough clusters!!!"
                    self.server.set_preempted(self.result)
                    self.is_processing = False
                    return
            elif num_clusters % 4 != 0:
                self.feedback.status = "Incorrect clustering, maybe another object detected. Check parameters!!!"
                self.feedback.success = False
                self.server.publish_feedback(self.feedback)
                if self.server.is_preempt_requested():
                    self.result.status = "Incorrect clustering was done, check parameters!!!"
                    self.server.set_preempted(self.result)
                    self.is_processing = False
                    return
            else:
                kmeans, sorted_clusters = utils.kmeans_clustering(clusters, num_clusters // 4)
                dolly_poses = utils.calculate_dolly_poses(
                    kmeans,
                    sorted_clusters,
                    self.dolly_dimension_tolerance,
                    self.dolly_dimensions
                )
                utils.publish_transforms(dolly_poses, sorted_clusters)
                respond = utils.generate_PoseArray(dolly_poses)

                self.feedback.status = f"{int(num_clusters/4)} dolly found."
                self.feedback.poses = respond
                self.feedback.success = True
                self.server.publish_feedback(self.feedback)
                if self.server.is_preempt_requested():
                    self.result.success = True
                    self.result.status = f"{int(num_clusters/4)} dolly found."
                    self.server.set_preempted(self.result)
                    self.is_processing = False
                    return

            if self.server.is_preempt_requested():
                self.result.success = True
                self.result.status = "Dolly pose estimation paused."
                self.server.set_preempted(self.result)
                self.is_processing = False

            if self.loop_rate_param > 0:
                self.loop_rate.sleep()

def main():
    rospy.init_node('dolly_pose_estimation_server')
    dolly_server = DollyPoseEstimationServer()
    rospy.spin()

if __name__ == '__main__':
    main()
