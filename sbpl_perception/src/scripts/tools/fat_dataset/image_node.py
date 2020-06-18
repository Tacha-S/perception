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
import pcl
from plyfile import PlyData, PlyElement
from utils import *
import argparse
import pprint
import yaml
# def image_info_cloud_callback(image_msg, camera_info, cloud_in):
    # print("Received image_info_cloud_callback in sync message")
# 
# def image_cloud_callback(image_msg, cloud_in):
    # print("Received image_cloud_callback in sync message")

class ImageNode(object):
    def __init__(self, config):
        self.config = config
        self.pp = pprint.PrettyPrinter(indent=4)
        self.pp.pprint(config)

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
        self.depth_scale = 100.0
        
        self.depth_image_cache = message_filters.Cache(depth_image_sub, 10000, allow_headerless=False)
        self.cv_bridge = CvBridge()
        self.output_path = config['label_gen']['output_path']
        model_path = get_model_path(config['label_gen']['model_dir'], config['label_gen']['object_name'], model_type="upright")
        self.initial_rotation = get_initial_rotation(config['label_gen']['object_label'])

        # model_path = "/media/aditya/A69AFABA9AFA85D9/Datasets/YCB_Video_Dataset/models/004_sugar_box/textured.ply"
        # self.output_path = "./bag_output/sugar_1"
        # self.initial_rotation = tf.transformations.quaternion_from_euler(0, 0, -math.pi/2) #sugar_1
        # self.output_path = "./bag_output/sugar_2"
        # self.initial_rotation = tf.transformations.quaternion_from_euler(0, 0, -2.0 * math.pi/3) #sugar_2
        # self.output_path = "./bag_output/sugar_3"
        # self.initial_rotation = tf.transformations.quaternion_from_euler(0, 0, 2.0 * math.pi/3) #sugar_3


        # model_path = "/media/aditya/A69AFABA9AFA85D9/Datasets/YCB_Video_Dataset/models/035_power_drill/textured_upright.ply"
        # self.output_path = "./bag_output/drill_1"
        # self.initial_rotation = tf.transformations.quaternion_from_euler(0, 0, math.pi) #drill_1
        # self.output_path = "./bag_output/drill_2"
        # self.initial_rotation = tf.transformations.quaternion_from_euler(0, 0, 2.5/3 * math.pi) #drill_2
        # self.output_path = "./bag_output/drill_3"
        # self.initial_rotation = tf.transformations.quaternion_from_euler(0, 0, -2.5/3 * math.pi) #drill_3

        # model_path = "/media/aditya/A69AFABA9AFA85D9/Datasets/YCB_Video_Dataset/models/005_tomato_soup_can/textured.ply"
        # self.output_path = "./bag_output/soup_1"
        # self.initial_rotation = tf.transformations.quaternion_from_euler(0, 0, 0) #soup_1

        # model_path = "/media/aditya/A69AFABA9AFA85D9/Datasets/YCB_Video_Dataset/models/006_mustard_bottle/textured.ply"
        # self.output_path = "./bag_output/mustard_1"
        # self.initial_rotation = tf.transformations.quaternion_from_euler(0, 0, 0) #mustard_1
        # self.output_path = "./bag_output/mustard_2"
        # self.initial_rotation = tf.transformations.quaternion_from_euler(0, 0, 2.5/3 * math.pi) #mustard_2
        # self.output_path = "./bag_output/sugar_3"
        # self.initial_rotation = tf.transformations.quaternion_from_euler(0, 0, 2.0 * math.pi/3) #sugar_3

        mkdir_if_missing(self.output_path)
        self.world_frame =  "/base_footprint"
        self.camera_frame =  "/camera_rgb_optical_frame"

        self.tf_listener = tf.TransformListener()
        self.camera_pose = None
        self.rgb_camera_instrinc_matrix = None
        self.depth_camera_instrinc_matrix = None
        self.counter = 0
        self.MAX_COUNTER = 400
        self.pub_filtered_cloud = \
            rospy.Publisher(
                "image_node/filtered_cloud",
                PointCloud2,
                queue_size=10
            )
        self.pub_pose_cloud = \
            rospy.Publisher(
                "image_node/pose_cloud",
                PointCloud2,
                queue_size=10
            )
        
        # Read Model
        cloud = PlyData.read(model_path).elements[0].data
        cloud = np.transpose(np.vstack((cloud['x'], cloud['y'], cloud['z'])))
        
        cloud_pose = pcl.PointCloud()
        cloud_pose.from_array(cloud)
        sor = cloud_pose.make_voxel_grid_filter()
        sor.set_leaf_size(0.015, 0.015, 0.015)
        cloud_pose = sor.filter()
        self.mesh_cloud = np.asarray(cloud_pose)
        # Z point up
        self.object_height = np.max(self.mesh_cloud[:,2]) - np.min(self.mesh_cloud[:,2])
        self.cloud_filter_params = {
            "xmin" : 0.1,
            "xmax" : 0.6,
            "ymin" : 0.0,
            "ymax" : 1.7,
            "zmin" : 0.75, #drill, sugar, mustard
            # "zmin" : 0.76, #soup can
            "object_height" : self.object_height,
            "downsampling_leaf_size" : 0.015
        }
        print(self.cloud_filter_params)

        print ("Num points after downsample and filter : {}".format(
            self.mesh_cloud.shape[0]))

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
            # Matrix needed for labelling using ICP
            np.savetxt(os.path.join(self.output_path, 'camera_pose_matrix.txt'), self.camera_pose_matrix)
            # JSON required by fat_pose_image.py
            with open(os.path.join(self.output_path, 'camera_pose.json'), 'w') as json_file:
                json.dump(camera_pose, json_file)
            return camera_pose
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.logerr("Couldnt get camera pose")
        return None

    def image_callback(self, image_msg):
        if self.camera_pose is None:
            self.pool = Pool(processes=8)              #Start a worker processes.
            self.camera_pose = self.get_camera_pose()

        if self.counter > self.MAX_COUNTER :
            return

        print("Received image_callback in message")
        latest_depth_image_msg = self.depth_image_cache.getElemBeforeTime(image_msg.header.stamp)   
        if latest_depth_image_msg is not None:
            print("Depth image msg received at timestamp {}".format(image_msg.header.stamp))
            rgb_img = self.cv_bridge.imgmsg_to_cv2(image_msg, "bgr8")
            depth_img = self.cv_bridge.imgmsg_to_cv2(latest_depth_image_msg)
            depth_img = depth_img * self.depth_scale
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
                # depth_img_registered, point_cloud = depth_rgb_registration(depth_img, 
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
                                                              self.depth_scale,
                                                              depth_registered_file))

            self.counter += 1

    def label_images(self):
        print("Deleting existing mask and pose labels")
        delete_from_dir(self.output_path, ".pose.txt")
        delete_from_dir(self.output_path, ".mask.jpg")
        
        print("Initial rotation being used : {}".format(self.initial_rotation))

        camera_pose_matrix_path = self.output_path + "/camera_pose_matrix.txt"    
        self.camera_pose_matrix = np.loadtxt(camera_pose_matrix_path)
        print("Camera pose : {}".format(self.camera_pose_matrix))
        camera_intrinsics_matrix_path = self.output_path + "/rgb_camera_intrinsics.txt"
        self.rgb_camera_instrinc_matrix = np.loadtxt(camera_intrinsics_matrix_path)
        print("RGB Camera intrinsics : {}".format(self.rgb_camera_instrinc_matrix))

        end_i = self.MAX_COUNTER

        # start_i = 150 #drill_1
        # start_i = 125 #drill_2, drill_3, sugar_2, sugar_1, sugar_3
        # start_i = 100 #soup_1
        # start_i = 150 #mustard_1, mustard_2, mustard_3
        start_i = self.config['label_gen']['start_index']

        pose_icp_prev = {}
        for img_i in np.arange(start_i, end_i, 1):
            # Read registered depth image
            depth_img_path = os.path.join(self.output_path, str(img_i) + ".depth.jpg")
            print("Labelling depth image : {}".format(depth_img_path))
            depth_data = cv2.imread(depth_img_path, cv2.IMREAD_UNCHANGED)
            cloud_in = image_to_cloud(depth_data, self.depth_scale, self.rgb_camera_instrinc_matrix)
            cloud_in_filtered, pose_estimate = process_cloud(cloud_in, 
                                                            self.camera_pose_matrix, 
                                                            self.pub_filtered_cloud,
                                                            self.initial_rotation,
                                                            self.cloud_filter_params)
            cloud_in_filtered_array = np.asarray(cloud_in_filtered)
            
            if cloud_in_filtered_array.shape[0] < 100:
                continue

            # Label Pose
            if img_i == start_i:
                pose_estimate_icp = pose_estimate
            else:
                print("Using previous ICP as input to next ICP")
                pose_estimate_icp = pose_icp_prev
            pose_icp_prev, pose_output_cam = do_icp(cloud_in_filtered,
                                                    self.camera_pose_matrix,
                                                    self.pub_filtered_cloud,
                                                    self.mesh_cloud,
                                                    self.pub_pose_cloud,
                                                    pose_estimate_icp)

            # Write Pose
            pose_output_path = os.path.join(self.output_path, str(img_i) + ".pose.txt")
            # np.savetxt(pose_output_path, pose_output_cam['matrix'])
            np.savetxt(pose_output_path, pose_icp_prev['matrix'])

            # Generate Mask Using Filtered Cloud
            cloud_in_filtered_cam = transform_cloud(cloud_in_filtered_array, 
                                                    mat=np.linalg.inv(self.camera_pose_matrix))
            mask_image = cloud_to_image(depth_data.shape[0], 
                                        depth_data.shape[1], 
                                        self.depth_scale, 
                                        cloud_in_filtered_cam,
                                        self.rgb_camera_instrinc_matrix)
            output_mask_path = os.path.join(self.output_path, str(img_i) + ".mask.jpg")
            cv2.imwrite(output_mask_path, mask_image)

            # mask_img = filtered_depth_image[filtered_depth_image > 0]
            # filtered_depth_image[filtered_depth_image > 0] = 255
            # filtered_depth_image = filtered_depth_image.astype(np.uint8)
            # filtered_depth_image, contours, hierarchy = cv2.findContours(filtered_depth_image,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            # print(contours)
            # cv2.fillPoly(filtered_depth_image, pts =[contours], color=(255,255,255))



def main():

    parser = argparse.ArgumentParser(description="Video stream from the command line")
    parser.add_argument("--parse_rosbag", action="store_true")
    parser.add_argument("--label_images", action="store_true")
    parser.add_argument("--config", "-c", dest='config', type=str, required=True)
    args = parser.parse_args()

    with open(args.config, 'r') as cfg:
        config = yaml.load(cfg)
    
    # Initialize ROS node
    rospy.init_node('write_images', disable_signals=True)
    print("Init node")

    img_node = ImageNode(config)
    if args.label_images:
        img_node.label_images()
    
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
    # r = rospy.Rate(10) # 10hz
    # while not rospy.is_shutdown():
    #     rospy.spin()
    #     r.sleep()
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()
