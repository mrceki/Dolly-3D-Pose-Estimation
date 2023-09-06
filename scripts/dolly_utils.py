#!/usr/bin/env python3

import rospy
import tf2_ros
from geometry_msgs.msg import Point, TransformStamped
import math
from sklearn.cluster import DBSCAN, KMeans
import numpy as np
import tf
from geometry_msgs.msg import Point, TransformStamped, PoseArray, Pose

# Size of dolly
DOLLY_SIZE_X = 1.12895
DOLLY_SIZE_Y = 1.47598
DOLLY_SIZE_HYPOTENUSE = (DOLLY_SIZE_X ** 2 + DOLLY_SIZE_Y ** 2) ** 0.5

class LegPointCluster:
    def __init__(self):
        self.points = []
    
    def add_point(self, point):
        self.points.append(point)
    
    def get_center_point(self):
        center_point = Point()
        num_points = len(self.points)
        if num_points > 0:
            sum_x = sum_y = 0.0
            for point in self.points:
                sum_x += point.x
                sum_y += point.y
            center_point.x = sum_x / num_points
            center_point.y = sum_y / num_points
        return center_point

# Convert laserscan data to cartesian data
def cartesian_conversion(scan_data, scan_range):
    cartesian_points = []
    angle = scan_data.angle_min
    for range_value in scan_data.ranges:
        if not math.isnan(range_value) and range_value != 0.0 and range_value < scan_range:
            x = range_value * math.cos(angle)
            y = range_value * math.sin(angle)
            point = Point(x, y, 0.0)
            cartesian_points.append(point)
        angle += scan_data.angle_increment
    return cartesian_points

# Calculates distance between two points, used in filtering
def calculate_distance(cluster1, cluster2):
    x1, y1 = cluster1.get_center_point().x, cluster1.get_center_point().y
    x2, y2 = cluster2.get_center_point().x, cluster2.get_center_point().y
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance

def cluster_points(points, eps, min_samples, max_samples, dolly_dimension_tolerance, scan_range):
    # Convert to numpy array
    data = np.array([[point.x, point.y] for point in points])

    # Clustering with DBSCAN algorithm
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(data)

    labels = db.labels_

    # Clustering 
    unfiltered_clusters = []
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    for i in range(num_clusters):
        cluster = LegPointCluster()
        cluster_points = [points[j] for j in range(len(points)) if labels[j] == i]
        for point in cluster_points:
            cluster.add_point(point)
        if len(cluster_points) <= max_samples:  # If there are no more than 5 instances in the cluster
            unfiltered_clusters.append(cluster)

   # Clustering filter -> Must have at least 3 clusters at 1.92 distance and clusters closer than 5 meters
    clusters = []
    for cluster in unfiltered_clusters:
        x1, y1 = cluster.get_center_point().x, cluster.get_center_point().y
        near_clusters = []
        for other_cluster in unfiltered_clusters:
            if other_cluster != cluster:
                x2, y2 = other_cluster.get_center_point().x, other_cluster.get_center_point().y
                distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                if distance <= 1.80: #hypotenuse of dolly 
                    near_clusters.append(other_cluster)
        if len(near_clusters) >= 2 and cluster.get_center_point().x**2 + cluster.get_center_point().y**2 <= scan_range**2: # Check Here 
            clusters.append(cluster)

    # Must other clusters at given sizes
    filtered_clusters = []

    dimension_offset = dolly_dimension_tolerance
    dimension_ranges = [(DOLLY_SIZE_X, dimension_offset), (DOLLY_SIZE_Y, dimension_offset), (DOLLY_SIZE_HYPOTENUSE, dimension_offset)]

    for cluster in clusters:
        x1, y1 = cluster.get_center_point().x, cluster.get_center_point().y
        valid_distance_count = 0

        for other_cluster in clusters:
            if other_cluster != cluster:
                x2, y2 = other_cluster.get_center_point().x, other_cluster.get_center_point().y
                distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

                if any((dim - offset <= distance <= dim + offset) for dim, offset in dimension_ranges):
                    valid_distance_count += 1
        # !!!!!!!!!!!!!1Fix here for more than 3 valid distance count!!!!!!!!!!!!!!
        if valid_distance_count == 3:         
            filtered_clusters.append(cluster)

    return filtered_clusters

def dolly_check(num_clusters):
    if num_clusters % 4 != 0:
        rospy.logwarn("Number of clusters is not divisible by 4.") #FixMe
        return False
    
def dbscan_clustering(cartesian_points, max_samples, dolly_dimension_tolerance, eps, min_samples, scan_range):
    # DBSCAN Clustering hyperparameters
    eps = eps # Distance (m)
    min_samples = min_samples  # Minimum samples

    dbscan_clusters = cluster_points(cartesian_points, eps, min_samples, max_samples, dolly_dimension_tolerance, scan_range)
    
    return dbscan_clusters

def kmeans_clustering(clusters, dolly_count):
     # Apply k-means algorithm to group clusters
    kmeans_data = np.array([[cluster.get_center_point().x, cluster.get_center_point().y] for cluster in clusters])
    kmeans_fit = KMeans(n_clusters=dolly_count, random_state=0, n_init="auto").fit(kmeans_data)
    # labels = kmeans.labels_.tolist()

    sorted_kmeans_clusters = [[] for _ in range(dolly_count)]
    for j in range(dolly_count*4):
        sorted_kmeans_clusters[kmeans_fit.labels_[j]].append(clusters[j])
    return kmeans_fit, sorted_kmeans_clusters

def sort_dolly_legs(i,sorted_clusters):
    distance_of_dolly_legs = []
    for j in range(4):
        x, y = sorted_clusters[i][j].get_center_point().x, sorted_clusters[i][j].get_center_point().y
        distance = math.sqrt(x ** 2 + y ** 2)
        distance_of_dolly_legs.append(distance)
    return distance_of_dolly_legs

# def sort_dolly_legs(i,sorted_clusters):
#     distance_of_dolly_legs = []
#     if i <= len(sorted_clusters):
#         # if len(sorted_clusters[i]) < 4:
#         #     del sorted_clusters[i]  # Adım 1
    
#         #     # Tüm alt listeleri bir adı m geri kaydırma
#         #     for j in range(i + 1, len(sorted_clusters)):
#         #         sorted_clusters[j - 1] = sorted_clusters[j]
    
#         #     # Son elemanı listeden çıkar
#         #     sorted_clusters.pop()
#         print(f"i = {i}")
#         for j in range(len(sorted_clusters[i])):
#             # print(f"lenght = {len(sorted_clusters[i])} and i = {i}")
#             if sorted_clusters[i][j] is not None:
#                 x, y = sorted_clusters[i][j].get_center_point().x, sorted_clusters[i][j].get_center_point().y
#                 distance = math.sqrt(x ** 2 + y ** 2)

def compare_and_sort_legs(i,sorted_clusters, dolly_dimension_tolerance):

    coordinates = [sorted_clusters[i][j].get_center_point() for j in range(4)]
    x0, y0 = coordinates[0].x, coordinates[0].y
    x1, y1 = coordinates[1].x, coordinates[1].y
    x2, y2 = coordinates[2].x, coordinates[2].y
    x3, y3 = coordinates[3].x, coordinates[3].y

    distance_1 = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2) 
    distance_2 = math.sqrt((x2 - x0) ** 2 + (y2 - y0) ** 2) 
    distance_3 = math.sqrt((x3 - x0) ** 2 + (y3 - y0) ** 2) 

    offset = dolly_dimension_tolerance
    if DOLLY_SIZE_HYPOTENUSE - offset <= distance_1 <= DOLLY_SIZE_HYPOTENUSE + offset:
        sorted_clusters[i][1], sorted_clusters[i][3] = sorted_clusters[i][3], sorted_clusters[i][1]
    if DOLLY_SIZE_HYPOTENUSE - offset <= distance_2 <= DOLLY_SIZE_HYPOTENUSE + offset:
        sorted_clusters[i][2], sorted_clusters[i][3] = sorted_clusters[i][3], sorted_clusters[i][2]
        if distance_3 < distance_1:
            sorted_clusters[i][1], sorted_clusters[i][2] = sorted_clusters[i][2], sorted_clusters[i][1]
    else:
        if distance_2 < distance_1:
            sorted_clusters[i][2], sorted_clusters[i][1] = sorted_clusters[i][1], sorted_clusters[i][2]

    return sorted_clusters

def calculate_dolly_poses(kmeans, sorted_clusters, dolly_dimension_tolerance):
    dolly_poses = []
    for i in range(len(kmeans.cluster_centers_)):
        dolly_center = Point()
        dolly_center.x = kmeans.cluster_centers_[i][0] * -1
        dolly_center.y = kmeans.cluster_centers_[i][1] * -1

        sorted_clusters[i] = sorted(sorted_clusters[i], key=lambda x: sort_dolly_legs(i, sorted_clusters))
        sorted_clusters = compare_and_sort_legs(i,sorted_clusters, dolly_dimension_tolerance)

        # Center points of clusters
        x1, y1 = sorted_clusters[i][0].get_center_point().x, sorted_clusters[i][0].get_center_point().y
        x2, y2 = sorted_clusters[i][2].get_center_point().x, sorted_clusters[i][2].get_center_point().y
        
        dolly_yaw = math.atan2(y2 - y1, x2 - x1) - math.pi/2
        dolly_poses.append((dolly_center, dolly_yaw))

    return dolly_poses

def publish_transforms(dolly_poses, sorted_clusters):
    dolly_transforms = []
    cluster_transforms = []
    tf_broadcaster = tf2_ros.TransformBroadcaster()

    for i, (dolly_center, dolly_yaw) in enumerate(dolly_poses):
        # Dolly TF
        dolly_transform = TransformStamped()
        dolly_transform.header.stamp = rospy.Time.now()
        dolly_transform.header.frame_id = "base_link"
        dolly_transform.child_frame_id = f"dolly_{i}"
        dolly_transform.transform.translation.x = dolly_center.x * -1
        dolly_transform.transform.translation.y = dolly_center.y * -1
        dolly_transform.transform.translation.z = 0.0
        quaternion = tf.transformations.quaternion_from_euler(0, 0, dolly_yaw)
        dolly_transform.transform.rotation.x = quaternion[0]
        dolly_transform.transform.rotation.y = quaternion[1]
        dolly_transform.transform.rotation.z = quaternion[2]
        dolly_transform.transform.rotation.w = quaternion[3]
        dolly_transforms.append(dolly_transform)

        rospy.loginfo(f"Center of dolly{i} ({dolly_center.x}, {dolly_center.y})")

        # # Cluster TF
        # for j, cluster in enumerate(sorted_clusters[i]):
        #     cluster_center = cluster.get_center_point()
        #     cluster_transform = TransformStamped()
        #     cluster_transform.header.stamp = rospy.Time.now()
        #     cluster_transform.header.frame_id = "base_link"
        #     cluster_transform.child_frame_id = f"cluster_{i*4+j}"
        #     cluster_transform.transform.translation.x = cluster_center.x
        #     cluster_transform.transform.translation.y = cluster_center.y
        #     cluster_transform.transform.translation.z = 0.0
        #     cluster_transform.transform.rotation.x = 0.0
        #     cluster_transform.transform.rotation.y = 0.0
        #     cluster_transform.transform.rotation.z = 0.0
        #     cluster_transform.transform.rotation.w = 1.0
        #     cluster_transforms.append(cluster_transform)

    tf_broadcaster.sendTransform(dolly_transforms)
    tf_broadcaster.sendTransform(cluster_transforms)
    rospy.loginfo("Number of Dolly Groups: %d", len(dolly_poses))

def generate_poseArray(dolly_poses):
    dolly_poseArray = PoseArray()
    dolly_poseArray.header.frame_id = "base_link"
    dolly_poseArray.header.stamp = rospy.Time.now()

    for i, (dolly_center, dolly_yaw) in enumerate(dolly_poses):
        dollys = Pose()
        dollys.position.x      = dolly_center.x
        dollys.position.y      = dolly_center.y
        dollys.position.z      = 0
        quaternion = tf.transformations.quaternion_from_euler(0, 0, dolly_yaw)
        dollys.orientation.x   = quaternion[0]
        dollys.orientation.y   = quaternion[1]    
        dollys.orientation.z   = quaternion[2]
        dollys.orientation.w   = quaternion[3]
        dolly_poseArray.poses.append(dollys)

    return dolly_poseArray