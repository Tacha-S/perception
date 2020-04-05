#!/usr/bin/env python

# Copyright (c) 2018 NVIDIA Corporation. All rights reserved.
# This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
# https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode

"""
This file starts a ROS node to run DOPE, 
listening to an image topic and publishing poses.
"""

from __future__ import print_function

import cv2
import message_filters
import numpy as np
import resource_retriever
import rospy
import tf.transformations
from PIL import Image
from PIL import ImageDraw
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import PointCloud2, CameraInfo, Image as ImageSensor_msg
from std_msgs.msg import String
import os
import json

# def image_info_cloud_callback(image_msg, camera_info, cloud_in):
    # print("Received image_info_cloud_callback in sync message")
# 
# def image_cloud_callback(image_msg, cloud_in):
    # print("Received image_cloud_callback in sync message")

class ImageNode(object):
    def __init__(self):
        rgb_image_sub = message_filters.Subscriber(
            "/camera/rgb/image_color",
            ImageSensor_msg
        )
        rgb_info_sub = message_filters.Subscriber(
            "/camera/rgb/camera_info",
            CameraInfo
        )
        depth_info_sub = message_filters.Subscriber(
            "/camera/depth/camera_info",
            CameraInfo
        )
        depth_image_sub = message_filters.Subscriber(
            "/camera/depth/image",
            ImageSensor_msg
        )
        
        self.depth_image_cache = message_filters.Cache(depth_image_sub, 100, allow_headerless=False)
        self.cv_bridge = CvBridge()
        self.output_path = "./output/drill"
        self.world_frame =  "/base_footprint"
        # self.camera_frame =  "/camera_depth_frame"
        self.camera_frame =  "/camera_rgb_optical_frame"

        self.tf_listener = tf.TransformListener()
        self.camera_pose = None
        self.rgb_camera_instrinc_matrix = None
        self.depth_camera_instrinc_matrix = None
        self.counter = 0

        rgb_image_sub.registerCallback(self.image_callback)
        rgb_info_sub.registerCallback(self.rgb_info_callback)
        depth_info_sub.registerCallback(self.depth_info_callback)


    def rgb_info_callback(self, camera_info):
        print("Received rgb_info_callback message")
        if self.rgb_camera_instrinc_matrix is None:
            self.rgb_camera_instrinc_matrix = np.matrix(camera_info.K, dtype='float64')
            self.rgb_camera_instrinc_matrix.resize((3, 3))
            np.savetxt(os.path.join(self.output_path, 'rgb_camera_intrinsics.txt'), self.rgb_camera_instrinc_matrix)
    

    def depth_info_callback(self, camera_info):
        print("Received depth_info_callback message")
        if self.depth_camera_instrinc_matrix is None:
            self.depth_camera_instrinc_matrix = np.matrix(camera_info.K, dtype='float64')
            self.depth_camera_instrinc_matrix.resize((3, 3))
            np.savetxt(os.path.join(self.output_path, 'depth_camera_intrinsics.txt'), self.depth_camera_instrinc_matrix)


    def get_camera_pose(self):
        try:
            rospy.logdebug("Getting transform between : {} and {}".format(self.world_frame, self.camera_frame))
            trans, quat = self.tf_listener.lookupTransform(self.world_frame, self.camera_frame, rospy.Time(0))
            # Camera pose in world frame
            R = tf.transformations.quaternion_matrix(quat)
            T = tf.transformations.translation_matrix(trans)
            camera_pose_matrix = tf.transformations.concatenate_matrices(T, R)
            rospy.logwarn("Got camera pose : {}".format(camera_pose_matrix))
            loc = tf.transformations.translation_from_matrix(camera_pose_matrix)
            ori = tf.transformations.quaternion_from_matrix(camera_pose_matrix)
            camera_pose =  {
                'location_worldframe': loc.tolist(), 
                'quaternion_xyzw_worldframe': ori.tolist()
            }
            with open(os.path.join(self.output_path, 'camera_pose.json'), 'w') as json_file:
                json.dump(camera_pose, json_file)
            return camera_pose
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.logerr("Couldnt get camera pose")
        return None

    def image_callback(self, image_msg):
        if self.camera_pose is None:
            self.camera_pose = self.get_camera_pose()


        print("Received image_callback in message")
        latest_depth_image_msg = self.depth_image_cache.getElemBeforeTime(image_msg.header.stamp)   
        if latest_depth_image_msg is not None:
            print("Depth image msg received at timestamp {}".format(image_msg.header.stamp))
            rgb_img = self.cv_bridge.imgmsg_to_cv2(image_msg, "bgr8")
            depth_img = self.cv_bridge.imgmsg_to_cv2(latest_depth_image_msg)
            depth_img = depth_img * 100
            print(depth_img)
            print(np.nanmax(depth_img))
            depth_img = np.asarray(depth_img, dtype=np.uint8)
            print(depth_img)
            image_file = os.path.join(self.output_path, str(self.counter) + ".color.jpg")
            depth_file = os.path.join(self.output_path, str(self.counter) + ".depth.jpg")

            cv2.imwrite(image_file, rgb_img)
            cv2.imwrite(depth_file, depth_img)

            self.counter += 1

def main():

    # Initialize ROS node
    rospy.init_node('write_images')
    print("Init node")
    ImageNode()
    
    # global cv_bridge
    # global depth_image_cache

    # cv_bridge = CvBridge()
    # depth_image_cache = message_filters.Cache(depth_image_sub, 100, allow_headerless=False)
    # rgb_image_sub.registerCallback(image_callback)
    # ts = message_filters.TimeSynchronizer([rgb_image_sub, info_sub, depth_image_sub], 1000)
    # # ts = message_filters.ApproximateTimeSynchronizer([image_sub, info_sub, cloud_sub], 1, 10, 0.1)

    # ts.registerCallback(image_info_cloud_callback)

    # ts = message_filters.TimeSynchronizer([image_sub, cloud_sub], 1000)
    # # ts = message_filters.ApproximateTimeSynchronizer([image_sub, cloud_sub], 1, 10, 0.1)

    # ts.registerCallback(image_cloud_callback)
    
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()
