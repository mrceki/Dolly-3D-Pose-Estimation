#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import LaserScan
import dolly_utils as utils

def scan_callback(scan_data):
    cartesian_points = utils.cartesian_conversion(scan_data)
    clusters = utils.dbscan_clustering(cartesian_points)
    # Check if the number of clusters is divisible by 4
    num_clusters = len(clusters)
    utils.dolly_check(num_clusters)
    if num_clusters < 4:
        return
    kmeans, sorted_clusters = utils.kmeans_clustering(clusters, num_clusters // 4)
   
    dolly_poses = utils.calculate_dolly_poses(kmeans, sorted_clusters)

    utils.publish_transforms(dolly_poses, sorted_clusters)

def main():
    rospy.init_node('dolly_pose_estimation')
    rospy.Subscriber('/scan', LaserScan, scan_callback, queue_size = 1)
    rospy.spin()

if __name__ == '__main__':
    main()