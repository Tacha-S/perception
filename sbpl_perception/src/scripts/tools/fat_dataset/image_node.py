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
from multiprocessing import Pool

# def image_info_cloud_callback(image_msg, camera_info, cloud_in):
    # print("Received image_info_cloud_callback in sync message")
# 
# def image_cloud_callback(image_msg, cloud_in):
    # print("Received image_cloud_callback in sync message")

class ImageNode(object):
    def __init__(self):
        rgb_image_sub = message_filters.Subscriber(
            "/camera/rgb/image_color",
            ImageSensor_msg,
            queue_size=10000
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
            ImageSensor_msg,
            queue_size=10000
        )
        
        self.depth_image_cache = message_filters.Cache(depth_image_sub, 10000, allow_headerless=False)
        self.cv_bridge = CvBridge()
        self.output_path = "./bag_output/sugar"
        self.world_frame =  "/base_footprint"
        # self.camera_frame =  "/camera_depth_frame"
        self.camera_frame =  "/camera_rgb_optical_frame"

        self.tf_listener = tf.TransformListener()
        self.camera_pose = None
        self.rgb_camera_instrinc_matrix = None
        self.depth_camera_instrinc_matrix = None
        self.counter = 0
        self.pool = Pool(processes=8)              # Start a worker processes.
        self.MAX_COUNTER = 400


        rgb_image_sub.registerCallback(self.image_callback)
        rgb_info_sub.registerCallback(self.rgb_info_callback)
        depth_info_sub.registerCallback(self.depth_info_callback)


    def rgb_info_callback(self, camera_info):
        if self.rgb_camera_instrinc_matrix is None:
            print("Received rgb_info_callback message")
            self.rgb_camera_instrinc_matrix = np.matrix(camera_info.K, dtype='float64')
            self.rgb_camera_instrinc_matrix.resize((3, 3))
            np.savetxt(os.path.join(self.output_path, 'rgb_camera_intrinsics.txt'), self.rgb_camera_instrinc_matrix)
    

    def depth_info_callback(self, camera_info):
        if self.depth_camera_instrinc_matrix is None:
            print("Received depth_info_callback message")
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
            self.camera_pose_matrix = camera_pose_matrix
            with open(os.path.join(self.output_path, 'camera_pose.json'), 'w') as json_file:
                json.dump(camera_pose, json_file)
            return camera_pose
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.logerr("Couldnt get camera pose")
        return None

    def image_callback(self, image_msg):
        if self.camera_pose is None:
            self.camera_pose = self.get_camera_pose()

        if self.counter > self.MAX_COUNTER :
            return

        print("Received image_callback in message")
        latest_depth_image_msg = self.depth_image_cache.getElemBeforeTime(image_msg.header.stamp)   
        if latest_depth_image_msg is not None:
            print("Depth image msg received at timestamp {}".format(image_msg.header.stamp))
            rgb_img = self.cv_bridge.imgmsg_to_cv2(image_msg, "bgr8")
            depth_img = self.cv_bridge.imgmsg_to_cv2(latest_depth_image_msg)
            depth_img = depth_img * 100
            # print(depth_img)
            # print(np.nanmax(depth_img))
            depth_img = np.asarray(depth_img, dtype=np.uint8)
            # print(depth_img)
            image_file = os.path.join(self.output_path, str(self.counter) + ".color.jpg")
            depth_file = os.path.join(self.output_path, str(self.counter) + ".unregistered_depth.jpg")
            depth_registered_file = os.path.join(self.output_path, str(self.counter) + ".depth.jpg")

            cv2.imwrite(image_file, rgb_img)
            cv2.imwrite(depth_file, depth_img)

            if self.rgb_camera_instrinc_matrix is not None and self.depth_camera_instrinc_matrix is not None:
                depth_img_registered = self.pool.apply_async(depth_rgb_registration, (depth_img, 
                                                              rgb_img, 
                                                              self.depth_camera_instrinc_matrix[0,0],
                                                              self.depth_camera_instrinc_matrix[1,1],
                                                              self.depth_camera_instrinc_matrix[0,2],
                                                              self.depth_camera_instrinc_matrix[1,2],
                                                              self.rgb_camera_instrinc_matrix[0,0],
                                                              self.rgb_camera_instrinc_matrix[1,1],
                                                              self.rgb_camera_instrinc_matrix[0,2],
                                                              self.rgb_camera_instrinc_matrix[1,2],
                                                              self.camera_pose_matrix,
                                                              100.0,
                                                              depth_registered_file))
            self.counter += 1


def depth_rgb_registration(depthData, rgbData,
                    fx_d, fy_d, cx_d, cy_d,
                    fx_rgb, fy_rgb, cx_rgb, cy_rgb,
                    extrinsics,
                    depthScale,
                    depth_registered_file):
 
    depthHeight = depthData.shape[0]
    depthWidth =  depthData.shape[1]
    depth_img_registered = np.zeros(depthData.shape)
    # Aligned will contain X, Y, Z, R, G, B values in its planes
    aligned = np.zeros((depthHeight, depthWidth, 3))
 
    for v in range(1, (depthHeight)):
        for u in range(1, (depthWidth)):
            #  Apply depth intrinsics
            z = depthData[v,u]/depthScale
            x = ((u - cx_d) * z) / fx_d
            y = ((v - cy_d) * z) / fy_d
            
            #  Apply the extrinsics
            # transformed = np.transpose(np.matmul(extrinsics, np.array([x, y, z, 1])))
            # aligned[v,u,1] = transformed[1]
            # aligned[v,u,2] = transformed[2]
            # aligned[v,u,3] = transformed[3]
            aligned[v,u,0] = x
            aligned[v,u,1] = y
            aligned[v,u,2] = z
 
    for v in range(1, (depthHeight)):
        for u in range(1, (depthWidth)):
            #  Apply RGB intrinsics
            x = (aligned[v,u,0] * fx_rgb / aligned[v,u,2]) + cx_rgb
            y = (aligned[v,u,1] * fy_rgb / aligned[v,u,2]) + cy_rgb
            
            # "x" and "y" are indices into the RGB frame, but they may contain
            # invalid values (which correspond to the parts of the scene not visible
            # to the RGB camera.
            # Do we have a valid index?
            if x > depthWidth or y > depthHeight or x < 10 or y < 0 or np.isnan(x) or np.isnan(y):
                continue
            
            #  Need some kind of interpolation. I just did it the lazy way
            x = int(round(x))
            y = int(round(y))
            # print("x:{}, y:{}, depth : {}".format(x,y,aligned[v,u,2]))
 
            depth_img_registered[y,x] = aligned[v,u,2] * depthScale
            # aligned(v,u,4) = single(rgbData(y, x, 1);
            # aligned(v,u,5) = single(rgbData(y, x, 2);
            # aligned(v,u,6) = single(rgbData(y, x, 3);
    print("Registered depth")
    cv2.imwrite(depth_registered_file, depth_img_registered)
    return depth_img_registered
    # def get_world_point(point, camera_intrinsic_matrix) :
    #     camera_fx_reciprocal_ = 1.0 / camera_intrinsic_matrix[0, 0]
    #     camera_fy_reciprocal_ = 1.0 / camera_intrinsic_matrix[1, 1]

    #     world_point = np.zeros(3)

    #     world_point[2] = point[2]
    #     world_point[0] = (point[0] - camera_intrinsic_matrix[0,2]) * point[2] * (camera_fx_reciprocal_)
    #     world_point[1] = (point[1] - camera_intrinsic_matrix[1,2]) * point[2] * (camera_fy_reciprocal_)

    #     return world_point

    # def get_image_point(world_point, camera_intrinsic_matrix):
    #     x = (aligned(v,u,1) * fx_rgb / aligned(v,u,3)) + cx_rgb
    #     y = (aligned(v,u,2) * fy_rgb / aligned(v,u,3)) + cy_rgb
    #     x = round(x)
    #     y = round(y)

    # def register_depth_image(self, depth_img, depth_camera_intrinsic_matrix, rgb_camera_instrinc_matrix):
    #     '''
    #         @depth_img - np.array with actual depth values for every pixel
    #     '''
    #     depth_img_registered = np.zeros(depth_img.shape)
    #     points_3d = np.zeros((depth_img.shape[0] * depth_img.shape[1], 3), dtype=np.float32)
    #     count = 0
    #     for x in range(depth_img.shape[1]):
    #         for y in range(depth_img.shape[0]):
    #             point = np.array([x,y,depth_img[y,x]])
    #             w_point = get_world_point(point)
    #             points_3d[count, :] = w_point.tolist()
    #             count += 1

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
