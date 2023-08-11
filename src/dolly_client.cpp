#include <ros/ros.h>
#include <dolly_pose_estimation/DollyPose.h>
#include <sensor_msgs/LaserScan.h>

int main(int argc, char** argv) {
    ros::init(argc, argv, "dolly_pose_estimation_client");
    ros::NodeHandle nh;

    ros::ServiceClient client = nh.serviceClient<dolly_pose_estimation::DollyPose>("/dolly_pose_estimation");
    dolly_pose_estimation::DollyPose srv;
    
    sensor_msgs::LaserScan scan_data;
    scan_data = *(ros::topic::waitForMessage<sensor_msgs::LaserScan>("/scan", nh));
    srv.request.scan_data = scan_data;

    if (client.call(srv)) 
    {
        std::vector<geometry_msgs::Pose> dolly_positions = srv.response.poses.poses;
        
        for (size_t i = 0; i < dolly_positions.size(); ++i) {
            geometry_msgs::Pose pose = dolly_positions[i];
            ROS_INFO("Dolly %zu: X = %f, Y = %f, Yaw = %f", i, pose.position.x, pose.position.y, pose.orientation.z);
        }
    } 
    else 
    {
        ROS_ERROR("Service call failed");
    }
    return 0;
}
