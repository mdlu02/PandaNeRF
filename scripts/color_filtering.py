#!/usr/bin/env python

import rospy
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header
from sensor_msgs.msg import PointField
import numpy as np

class KinectFilterNode:

    def __init__(self):
        rospy.init_node('kinect_filter_node')

        # Subscribe to the Kinect point cloud topic
        rospy.Subscriber('/camera/depth/color/points', PointCloud2, self.kinect_callback)

        # Create a publisher to publish the filtered point cloud
        self.pub = rospy.Publisher('/camera/yellow_points', PointCloud2, queue_size=10)

    def kinect_callback(self, point_cloud):
        # Convert the point cloud to a numpy array
        point_cloud_array = np.array(list(pc2.read_points(point_cloud, skip_nans=True)))

        # Filter the point cloud by the color yellow
        yellow_points = point_cloud_array[np.where((point_cloud_array[:, 3] > 200) & (point_cloud_array[:, 4] > 200) & (point_cloud_array[:, 5] < 100))]

        # Create a new PointCloud2 message for the filtered point cloud
        filtered_cloud_msg = PointCloud2()
        filtered_cloud_msg.header = Header()
        filtered_cloud_msg.header.stamp = rospy.Time.now()
        filtered_cloud_msg.header.frame_id = point_cloud.header.frame_id
        filtered_cloud_msg.height = 1
        filtered_cloud_msg.width = yellow_points.shape[0]
        filtered_cloud_msg.fields.append(PointField(name="x", offset=0, datatype=7, count=1))
        filtered_cloud_msg.fields.append(PointField(name="y", offset=4, datatype=7, count=1))
        filtered_cloud_msg.fields.append(PointField(name="z", offset=8, datatype=7, count=1))
        filtered_cloud_msg.fields.append(PointField(name="rgb", offset=16, datatype=7, count=1))
        filtered_cloud_msg.point_step = 32
        filtered_cloud_msg.row_step = filtered_cloud_msg.point_step * filtered_cloud_msg.width
        filtered_cloud_msg.is_dense = True
        filtered_cloud_msg.is_bigendian = False
        filtered_cloud_msg.data = yellow_points.tostring()

        # Publish the filtered point cloud
        self.pub.publish(filtered_cloud_msg)

if __name__ == '__main__':
    try:
        node = KinectFilterNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


