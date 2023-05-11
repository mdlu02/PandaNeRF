#!/usr/bin/env python

import rospy
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import TransformBroadcaster

from realsense_camera_node.msg import RealSenseData

# Node initialization
rospy.init_node("realsense_camera_node")

# Publisher
realsense_data_pub = rospy.Publisher(
    "/camera/realsense_data", RealSenseData, queue_size=10
)


# Publishers
rgb_pub = rospy.Publisher("/camera/rgb/image_raw", Image, queue_size=10)
depth_pub = rospy.Publisher("/camera/depth/image_raw", Image, queue_size=10)
camera_info_pub = rospy.Publisher("/camera/camera_info", CameraInfo, queue_size=10)
pose_pub = rospy.Publisher("/camera/pose", TransformStamped, queue_size=10)

# Transform broadcaster
br = TransformBroadcaster()


def rgb_callback(rgb_image):
    rgb_pub.publish(rgb_image)


def depth_callback(depth_image):
    depth_pub.publish(depth_image)


def camera_info_callback(camera_info):
    camera_info_pub.publish(camera_info)


def pose_callback(pose):
    pose_pub.publish(pose)


def realsense_callback(rgb_image, depth_image, camera_info, pose):
    realsense_data = RealSenseData()
    realsense_data.rgb_image = rgb_image
    realsense_data.depth_image = depth_image
    realsense_data.camera_info = camera_info
    realsense_data.pose = pose
    realsense_data_pub.publish(realsense_data)


if __name__ == "__main__":
    # Subscribe to the RealSense camera data
    rospy.Subscriber("/camera/color/image_raw", Image, rgb_callback)
    rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, depth_callback)
    rospy.Subscriber("/camera/color/camera_info", CameraInfo, camera_info_callback)

    # If you have a method to estimate the camera pose, create a subscriber to receive that data
    rospy.Subscriber("/camera/estimated_pose", TransformStamped, pose_callback)

    # Spin the node
    rospy.spin()
