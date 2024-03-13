#!/usr/bin/env python3

import rospy

from dolly_pose_estimation.srv import DollyPoseEstimate
import dolly_utils as utils

# Size of dolly
DOLLY_SIZE_X = 1.12895
DOLLY_SIZE_Y = 1.47598
DOLLY_SIZE_HYPOTENUSE = (DOLLY_SIZE_X ** 2 + DOLLY_SIZE_Y ** 2) ** 0.5

dolly_dimensions = [DOLLY_SIZE_X, DOLLY_SIZE_Y]
dolly_dimension_tolerance = 0.15

def convert_request(request):
    clusters = []
    for pose in request.cluster_poses.poses:
        cluster = utils.LegPointCluster()  # Instantiate a new LegPointCluster object
        cluster.add_point(pose.position)  # Add the pose position to the cluster
        clusters.append(cluster)  # Append the cluster object to the clusters list
    return clusters

def dolly_pose_estimation_service(request):
    # cartesian_points = utils.cartesian_conversion(request.cluster_poses)
    # clusters = utils.dbscan_clustering(cartesian_points)
    clusters = convert_request(request)
    # Check if the number of clusters is divisible by 4
    num_clusters = len(clusters)
    utils.dolly_check(num_clusters)
    if num_clusters < 4:
        return
    kmeans, sorted_clusters, kmeans_result = utils.kmeans_clustering(clusters, num_clusters // 4)

    dolly_poses = utils.calculate_dolly_poses(kmeans, sorted_clusters, dolly_dimension_tolerance, dolly_dimensions)
    utils.publish_transforms(dolly_poses, sorted_clusters)
    respond = utils.generate_PoseArray(dolly_poses)
    return respond

def dolly_pose_estimation_server():
    rospy.init_node('dolly_pose_estimation_server')
    rospy.Service('/dolly_pose_estimation', DollyPoseEstimate, dolly_pose_estimation_service)
    rospy.spin()

if __name__ == '__main__':
    dolly_pose_estimation_server()