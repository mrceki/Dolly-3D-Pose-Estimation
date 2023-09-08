#!/usr/bin/env python3

import rospy
import tf2_ros
from geometry_msgs.msg import Point, TransformStamped
import math
from sklearn.cluster import DBSCAN, KMeans
import numpy as np
import tf
from geometry_msgs.msg import Point, TransformStamped, PoseArray, Pose

class LegPointCluster:
    """
    The `LegPointCluster` class represents a cluster of laser points and provides methods to add points
    to the cluster and calculate the center point of the cluster.
    """
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

def cartesian_conversion(scan_data, scan_range):
    """
    The function takes scan data and a scan range, and converts the data from polar coordinates to
    Cartesian coordinates.
    
    :param scan_data: The scan_data parameter is assumed to be an object that contains information about
    a laser scan. It likely has attributes such as angle_min, angle_max, angle_increment, and ranges.
    The angle_min attribute represents the minimum angle of the laser scan, angle_max represents the
    maximum angle, angle_increment represents the
    :param scan_range: The scan_range parameter represents the maximum range value for the scan data. It
    is used to filter out range values that are greater than this value
    :return: a list of cartesian points.
    """
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

def calculate_distance(cluster1, cluster2):
    """
    The function calculates the distance between the center points of two clusters.
    
    :param cluster1: The first cluster object
    :param cluster2: The `cluster2` parameter is the second cluster object that you want to calculate
    the distance to
    :return: the distance between the center points of two clusters.
    """
    x1, y1 = cluster1.get_center_point().x, cluster1.get_center_point().y
    x2, y2 = cluster2.get_center_point().x, cluster2.get_center_point().y
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance

def cluster_points(points, eps, min_samples, max_samples):
    """
    The function `cluster_points` takes a list of points, an epsilon value, a minimum number of samples,
    and a maximum number of samples, and uses the DBSCAN algorithm to cluster the points based on their
    proximity, returning only the clusters that have a number of points within the specified range.
    
    :param points: The "points" parameter is a list of objects representing points in a 2-dimensional
    space. Each point object should have attributes "x" and "y" representing the coordinates of the
    point
    :param eps: The eps parameter in the cluster_points function is the maximum distance between two
    points for them to be considered as part of the same cluster. It determines the radius of the
    neighborhood around each point
    :param min_samples: The `min_samples` parameter is the minimum number of points required to form a
    cluster. Any cluster with fewer points than `min_samples` will be considered noise and will not be
    included in the final result
    :param max_samples: The `max_samples` parameter is the maximum number of points allowed in a
    cluster. Any cluster with more than `max_samples` points will be excluded from the final result
    :return: a list of clusters that meet the criteria of having a minimum number of points
    (min_samples) and a maximum number of points (max_samples).
    """
    # Convert to numpy array
    data = np.array([[point.x, point.y] for point in points])
    # Clustering with DBSCAN algorithm
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(data)

    labels = db.labels_

    clusters = {}
    for i, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = LegPointCluster()
        clusters[label].add_point(points[i])

    return [cluster for cluster in clusters.values() if len(cluster.points) >= min_samples and len(cluster.points)<= max_samples] 

def filter_clusters(clusters, dolly_dimension_tolerance, cluster_range, dolly_dimensions):
    """
    The function filters clusters based on their distance from each other and their center point's
    distance from the origin.
    
    :param clusters: The `clusters` parameter is a list of objects representing clusters. Each cluster
    object has a `get_center_point()` method that returns the center point of the cluster as an object
    with `x` and `y` attributes
    :param dolly_dimension_tolerance: The `dolly_dimension_tolerance` parameter is a value that
    represents the acceptable tolerance or margin of error for the dimensions of the dolly. It is used
    to determine if the distance between two clusters falls within the acceptable range based on the
    dimensions of the dolly
    :param cluster_range: The cluster_range parameter represents the maximum distance from the origin (0,0)
    that a cluster's center point can be in order to be considered valid
    :param dolly_dimensions: The `dolly_dimensions` parameter is a list containing the dimensions of the
    dolly. The first element of the list represents the size of the dolly in the x-direction, and the
    second element represents the size of the dolly in the y-direction
    :return: a list of filtered clusters.
    """
    filtered_clusters = []

    for cluster in clusters:
        x1, y1 = cluster.get_center_point().x, cluster.get_center_point().y
        valid_distance_count = 0

        for other_cluster in clusters:
            if other_cluster != cluster:
                x2, y2 = other_cluster.get_center_point().x, other_cluster.get_center_point().y
                distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                dimension_offset = dolly_dimension_tolerance
                DOLLY_SIZE_HYPOTENUSE = (dolly_dimensions[0] ** 2 + dolly_dimensions[1] ** 2) ** 0.5 # [0] = size of x, [1] = size of y
                dimension_ranges = [(dolly_dimensions[0], dimension_offset), (dolly_dimensions[1], dimension_offset), (DOLLY_SIZE_HYPOTENUSE, dimension_offset)]

                if any((dim - offset <= distance <= dim + offset) for dim, offset in dimension_ranges):
                    valid_distance_count += 1
        if valid_distance_count == 3 and cluster.get_center_point().x ** 2 + cluster.get_center_point().y ** 2 <= cluster_range ** 2:
            filtered_clusters.append(cluster)

    return filtered_clusters

def dolly_check(num_clusters):
    """
    The function checks if the number of clusters is divisible by 4 and logs a warning if it is not.
    
    :param num_clusters: The number of clusters to check
    :return: False.
    """
    if num_clusters % 4 != 0:
        rospy.logwarn("Number of clusters is not divisible by 4.") #FixMe
        return False
    
def dbscan_clustering(cartesian_points, max_samples, dolly_dimension_tolerance, eps, min_samples, cluster_range, dolly_dimensions):
    """
    The function performs DBSCAN clustering on a set of Cartesian points and filters the resulting
    clusters based on specified criteria.
    
    :param cartesian_points: A list of points in Cartesian coordinates. Each point is represented as a
    tuple of (x, y) coordinates
    :param max_samples: The maximum number of samples that can be included in a cluster
    :param dolly_dimension_tolerance: The dolly_dimension_tolerance parameter is a tolerance value that
    determines how close the points in a cluster need to be in terms of their dolly dimensions in order
    to be considered part of the same cluster
    :param eps: The eps parameter in DBSCAN (Density-Based Spatial Clustering of Applications with
    Noise) is the maximum distance between two samples for them to be considered as part of the same
    neighborhood. In other words, it defines the radius of the neighborhood around each point
    :param min_samples: The minimum number of samples required for a cluster to be considered valid
    :param scan_range: The scan_range parameter represents the range within which the clustering
    algorithm will search for neighboring points. It is used to determine the maximum distance between
    two points for them to be considered neighbors
    :param dolly_dimensions: The dolly_dimensions parameter is a list of dimensions that are specific to
    the dolly. These dimensions are used to filter out clusters that do not meet the specified tolerance
    in those dimensions
    :return: the filtered clusters.
    """
    clusters = cluster_points(cartesian_points, eps, min_samples, max_samples)
    filtered_clusters = filter_clusters(clusters, dolly_dimension_tolerance, cluster_range, dolly_dimensions)
    return filtered_clusters

def kmeans_clustering(clusters, dolly_count):
    """
    The function `kmeans_clustering` applies the k-means algorithm to group clusters and returns the
    k-means fit and sorted clusters.
    
    :param clusters: The "clusters" parameter is a list of cluster objects. Each cluster object
    represents a group of data points that are similar to each other. Each cluster object has a center
    point, which is the average of all the data points in that cluster
    :param dolly_count: The parameter "dolly_count" represents the number of clusters or groups that you
    want to create using the k-means algorithm. It determines how many distinct groups the data will be
    divided into
    :return: two values: `kmeans_fit`, which is the result of applying the k-means algorithm to the
    clusters, and `sorted_kmeans_clusters`, which is a list of lists containing the clusters sorted
    based on the k-means labels.
    """
     # Apply k-means algorithm to group clusters
    kmeans_data = np.array([[cluster.get_center_point().x, cluster.get_center_point().y] for cluster in clusters])
    kmeans_fit = KMeans(n_clusters=dolly_count, random_state=0, n_init="auto").fit(kmeans_data)

    sorted_kmeans_clusters = [[] for _ in range(dolly_count)]
    for j in range(dolly_count*4):
        sorted_kmeans_clusters[kmeans_fit.labels_[j]].append(clusters[j])
    return kmeans_fit, sorted_kmeans_clusters

def sort_dolly_legs(i,sorted_clusters):
    """
    The function calculates the distance of each leg of a Dolly object from the origin and returns a
    list of these distances.
    
    :param i: The parameter "i" represents the index of the cluster in the "sorted_clusters" list
    :param sorted_clusters: sorted_clusters is a list of clusters, where each cluster is a list of
    objects. Each object has a method get_center_point() that returns the center point of the object as
    a coordinate (x, y)
    :return: a list of distances of the center points of the dolly legs in the sorted_clusters.
    """
    distance_of_dolly_legs = []
    for j in range(4):
        x, y = sorted_clusters[i][j].get_center_point().x, sorted_clusters[i][j].get_center_point().y
        distance = math.sqrt(x ** 2 + y ** 2)
        distance_of_dolly_legs.append(distance)
    return distance_of_dolly_legs

def compare_and_sort_legs(i,sorted_clusters, dolly_dimension_tolerance, dolly_dimensions):
    """
    The function compares and sorts legs based on their distances from the center point and dolly
    dimensions.
    
    :param i: The parameter "i" is an integer representing the index of the cluster in the
    "sorted_clusters" list that we want to compare and sort the legs for
    :param sorted_clusters: The `sorted_clusters` parameter is a list of lists. Each inner list
    represents a cluster and contains four objects. Each object represents a leg and has a
    `get_center_point()` method that returns the coordinates of the leg's center point
    :param dolly_dimension_tolerance: The parameter "dolly_dimension_tolerance" is a value that
    represents the acceptable tolerance for the difference between the calculated distance of a leg and
    the expected distance for a dolly. It is used to determine if a leg needs to be rearranged in the
    sorted_clusters list
    :param dolly_dimensions: The `dolly_dimensions` parameter is a list containing the size of the dolly
    in the x and y dimensions. The first element of the list represents the size of the dolly in the x
    dimension, and the second element represents the size of the dolly in the y dimension
    :return: the sorted_clusters list after performing some comparisons and sorting operations on the
    legs.
    """

    coordinates = [sorted_clusters[i][j].get_center_point() for j in range(4)]
    x0, y0 = coordinates[0].x, coordinates[0].y
    x1, y1 = coordinates[1].x, coordinates[1].y
    x2, y2 = coordinates[2].x, coordinates[2].y
    x3, y3 = coordinates[3].x, coordinates[3].y

    distance_1 = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2) 
    distance_2 = math.sqrt((x2 - x0) ** 2 + (y2 - y0) ** 2) 
    distance_3 = math.sqrt((x3 - x0) ** 2 + (y3 - y0) ** 2) 

    DOLLY_SIZE_HYPOTENUSE = (dolly_dimensions[0] ** 2 + dolly_dimensions[1] ** 2) ** 0.5 # [0] = size of x, [1] = size of y
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

def calculate_dolly_poses(kmeans, sorted_clusters, dolly_dimension_tolerance, dolly_dimensions):
    """
    The function calculates the poses (center and yaw) of dolly objects based on k-means clustering and
    sorted clusters.
    
    :param kmeans: The `kmeans` parameter is a KMeans object that represents the result of clustering
    the data points. It contains information about the cluster centers and labels
    :param sorted_clusters: sorted_clusters is a list of lists, where each inner list represents a
    cluster of points. Each point in the cluster is an object with attributes x and y representing its
    coordinates
    :param dolly_dimension_tolerance: The dolly_dimension_tolerance parameter is a value that determines
    the tolerance for the difference in dimensions between the legs of the dolly. It is used to compare
    and sort the legs based on their dimensions
    :param dolly_dimensions: The `dolly_dimensions` parameter is a list that contains the dimensions of
    the dolly. It could be something like `[length, width, height]`, where `length` is the length of the
    dolly, `width` is the width of the dolly, and `height` is
    :return: a list of tuples, where each tuple contains a dolly center point (x, y coordinates) and a
    dolly yaw angle.
    """
    dolly_poses = []
    for i in range(len(kmeans.cluster_centers_)):
        dolly_center = Point()
        dolly_center.x = kmeans.cluster_centers_[i][0] * -1
        dolly_center.y = kmeans.cluster_centers_[i][1] * -1

        sorted_clusters[i] = sorted(sorted_clusters[i], key=lambda x: sort_dolly_legs(i, sorted_clusters))
        sorted_clusters = compare_and_sort_legs(i,sorted_clusters, dolly_dimension_tolerance, dolly_dimensions)

        x1, y1 = sorted_clusters[i][0].get_center_point().x, sorted_clusters[i][0].get_center_point().y
        x2, y2 = sorted_clusters[i][2].get_center_point().x, sorted_clusters[i][2].get_center_point().y
        
        dolly_yaw = math.atan2(y2 - y1, x2 - x1) - math.pi/2
        dolly_poses.append((dolly_center, dolly_yaw))

    return dolly_poses

def publish_transforms(dolly_poses, tf_flip):
    """
    The function `publish_transforms` publishes transforms for dolly poses and cluster poses using the
    `tf2_ros.TransformBroadcaster`.
    
    :param dolly_poses: The `dolly_poses` parameter is a list of tuples, where each tuple contains the
    position and yaw angle of a dolly. The position is represented by a `Point` object with `x` and `y`
    coordinates, and the yaw angle is a scalar value
    """
    dolly_transforms = []
    cluster_transforms = []
    tf_broadcaster = tf2_ros.TransformBroadcaster()
    
    for i, (dolly_center, dolly_yaw) in enumerate(dolly_poses):
        # Dolly TF
        dolly_transform = TransformStamped()
        dolly_transform.header.stamp = rospy.Time.now()
        dolly_transform.header.frame_id = "base_link"
        dolly_transform.child_frame_id = f"dolly_{i}"
        dolly_transform.transform.translation.x = dolly_center.x if not tf_flip else dolly_center.x * -1
        dolly_transform.transform.translation.y = dolly_center.y if not tf_flip else dolly_center.y * -1
        dolly_transform.transform.translation.z = 0.0
        quaternion = tf.transformations.quaternion_from_euler(0, 0, dolly_yaw)
        dolly_transform.transform.rotation.x = quaternion[0]
        dolly_transform.transform.rotation.y = quaternion[1]
        dolly_transform.transform.rotation.z = quaternion[2]
        dolly_transform.transform.rotation.w = quaternion[3]
        
        dolly_transforms.append(dolly_transform)

        rospy.loginfo(f"Center of dolly_{i} ({dolly_center.x}, {dolly_center.y})")

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

def generate_PoseArray(dolly_poses):
    """
    The function `generate_PoseArray` takes a list of dolly poses and converts them into a PoseArray
    message with appropriate header and pose values.
    
    :param dolly_poses: dolly_poses is a list of tuples, where each tuple contains the dolly center
    position (x, y) and the dolly yaw angle
    :return: a PoseArray object named "dolly_PoseArray".
    """
    dolly_PoseArray = PoseArray()
    dolly_PoseArray.header.frame_id = "base_link"
    dolly_PoseArray.header.stamp = rospy.Time.now()

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
        dolly_PoseArray.poses.append(dollys)

    return dolly_PoseArray