#!/usr/bin/env python3

import rospy

from dolly_pose_estimation.srv import DollyPose
import dolly_utils as utils

# Size of dolly
DOLLY_SIZE_X = 1.12895
DOLLY_SIZE_Y = 1.47598
DOLLY_SIZE_HYPOTENUSE = (DOLLY_SIZE_X ** 2 + DOLLY_SIZE_Y ** 2) ** 0.5

def dolly_pose_estimation_service(request):
    cartesian_points = utils.cartesian_conversion(request.scan_data)
    clusters = utils.dbscan_clustering(cartesian_points)

    # Check if the number of clusters is divisible by 4
    num_clusters = len(clusters)
    utils.dolly_check(num_clusters)
    if num_clusters < 4:
        return
    kmeans, sorted_clusters = utils.kmeans_clustering(clusters, num_clusters // 4)

    dolly_poses = utils.calculate_dolly_poses(kmeans, sorted_clusters)
    utils.publish_transforms(dolly_poses, sorted_clusters)
    respond = utils.enerate_PoseArray(dolly_poses)
    return respond

def dolly_pose_estimation_server():
    rospy.init_node('dolly_pose_estimation_server')
    rospy.Service('/dolly_pose_estimation', DollyPose, dolly_pose_estimation_service)
    rospy.spin()

if __name__ == '__main__':
    dolly_pose_estimation_server()