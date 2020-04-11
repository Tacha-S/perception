
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
import math

def get_initial_rotation(object_label):
    if object_label == "sugar_1":
        initial_rotation = tf.transformations.quaternion_from_euler(0, 0, -math.pi/2)
    elif object_label == "sugar_2":
        initial_rotation = tf.transformations.quaternion_from_euler(0, 0, -2.0 * math.pi/3)
    elif object_label == "sugar_3":
        initial_rotation = tf.transformations.quaternion_from_euler(0, 0, 2.0 * math.pi/3)
    elif object_label == "drill_1":
        initial_rotation = tf.transformations.quaternion_from_euler(0, 0, math.pi)
    elif object_label == "drill_2":
        initial_rotation = tf.transformations.quaternion_from_euler(0, 0, 2.5/3 * math.pi)
    elif object_label == "drill_3":
        initial_rotation = tf.transformations.quaternion_from_euler(0, 0, -2.5/3 * math.pi)
    elif object_label == "mustard_1":
        initial_rotation = tf.transformations.quaternion_from_euler(0, 0, 0)
    elif object_label == "mustard_2":
        initial_rotation = tf.transformations.quaternion_from_euler(0, 0, 2.5/3 * math.pi)
    elif object_label == "mustard_3":
        initial_rotation = tf.transformations.quaternion_from_euler(0, 0, -2.0 * math.pi/3)
    elif object_label == "soup_1":
        initial_rotation = tf.transformations.quaternion_from_euler(0, 0, 0)

    return initial_rotation

def get_model_path(model_dir, object_name, model_type="default"):
    '''
        Get absolute path of model from model dir
    '''
    if model_type == "default":
        model_path = os.path.join(model_dir, object_name, 'textured.ply')
    elif model_type == "upright":
        # For things like drill which need to be made upright
        temp_path = os.path.join(model_dir, object_name, 'textured_upright.ply')
        if os.path.exists(temp_path):
            model_path = temp_path
        else:
            model_path = os.path.join(model_dir, object_name, 'textured.ply')
    return model_path

def delete_from_dir(dir_name, extension):
    test = os.listdir(dir_name)
    for item in test:
        if item.endswith(extension):
            os.remove(os.path.join(dir_name, item))

def mkdir_if_missing(dst_dir):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
   
def transform_cloud(cloud_in, trans=None, quat=None, mat=None):
    '''
        Tranform point cloud np array
    '''
    if trans is not None and quat is not None:
        R = tf.transformations.quaternion_matrix(quat)
        T = tf.transformations.translation_matrix(trans)
        total_transform = tf.transformations.concatenate_matrices(T, R)
    elif mat is not None:
        total_transform = mat
    cloud_in = np.hstack((cloud_in, np.ones((cloud_in.shape[0], 1))))
    cloud_out = np.matmul(total_transform, np.transpose(cloud_in))
    cloud_out = np.transpose(cloud_out)[:,:3]
    cloud_out = np.array(cloud_out, dtype=np.float32)
    return cloud_out

def process_cloud(np_points, camera_matrix, pub_filtered_cloud, initial_rotation, filter_params, publish_filtered=True):
    '''
        1. Transform to robot frame
        2. Filter in z and x
        3. Downsample
        4. Publish and return filtered cloud
    '''
    # rospy.logdebug("Cloud received, frame : {}".format(cloud_msg.header.frame_id))

    # pc = ros_numpy.numpify(cloud_msg)
    # # pc_l = [np.asarray(x) for x in pc]
    # # print(pc_l)
    # # pc = np.asarray(pc_l)
    # # height = pc.shape[0]
    # # width = pc.shape[1]
    # # np_points = np.zeros((height * width, 3), dtype=np.float32)
    # np_points = np.zeros((pc.shape[0], 3), dtype=np.float32)
    # np_points[:, 0] = np.resize(pc['x'], pc.shape[0])
    # np_points[:, 1] = np.resize(pc['y'], pc.shape[0])
    # np_points[:, 2] = np.resize(pc['z'], pc.shape[0])
    # print(np_points.shape)
    # import pdb
    # pdb.set_trace()
    # np_points_appened = np.hstack((np_points, np.ones((np_points.shape[0], 1))))
    # np_points_transformed = np.matmul(total_transform, np.transpose(np_points_appened))
    # np_points_transformed = np.transpose(np_points_transformed)[:,:3]
    np_points_transformed = transform_cloud(np_points, mat=camera_matrix)
    pcl_cloud = pcl.PointCloud(np_points_transformed)

    passthrough = pcl_cloud.make_passthrough_filter()
    passthrough.set_filter_field_name("z")
    zmin = filter_params['zmin']
    passthrough.set_filter_limits(zmin, 1.0)
    cloud_filtered = passthrough.filter()

    passthrough = cloud_filtered.make_passthrough_filter()
    passthrough.set_filter_field_name("x")
    passthrough.set_filter_limits(filter_params['xmin'], filter_params['xmax'])
    cloud_filtered = passthrough.filter()

    passthrough = cloud_filtered.make_passthrough_filter()
    passthrough.set_filter_field_name("y")
    passthrough.set_filter_limits(filter_params['ymin'], filter_params['ymax'])
    cloud_filtered = passthrough.filter()

    # fil = cloud_filtered.make_statistical_outlier_filter()
    # fil.set_mean_k(100)
    # fil.set_std_dev_mul_thresh(0.1)
    # cloud_filtered = fil.filter()

    sor = cloud_filtered.make_voxel_grid_filter()
    sor.set_leaf_size(filter_params['downsampling_leaf_size'], filter_params['downsampling_leaf_size'], filter_params['downsampling_leaf_size'])
    cloud_filtered = sor.filter()

    fil = cloud_filtered.make_statistical_outlier_filter()
    fil.set_mean_k(100)
    fil.set_std_dev_mul_thresh(0.08)
    cloud_filtered = fil.filter()

    cloud_filtered_array = np.asarray(cloud_filtered)  
    # cloud_filtered_array = np_points  
    # cloud_filtered = pcl.PointCloud(np_points)

    mean = np.mean(cloud_filtered_array, axis=0)
    mean[2] = filter_params['zmin'] + filter_params['object_height']/2.0
    pose_estimate = {}
    # mean = mean
    pose_estimate['location'] = mean.tolist()
    # pose_estimate['quaternion'] = [0.02218474, -0.81609224, -0.57647473,  0.03432471]
    # pose_estimate['quaternion'] = [0,0,0,1]
    pose_estimate['quaternion'] = initial_rotation
    
    print ("Num points after downsample and filter in input cloud : {}".format(cloud_filtered_array.shape[0]))
    
    if publish_filtered:
        cloud_color = np.zeros(cloud_filtered_array.shape[0])
        ros_msg = xyzrgb_array_to_pointcloud2(
            cloud_filtered_array, cloud_color, rospy.Time.now(), "/base_footprint"
        )
        pub_filtered_cloud.publish(ros_msg)
    rospy.logdebug("Done cloud processing")
    
    return cloud_filtered, pose_estimate

def xyzrgb_array_to_pointcloud2(points, colors, stamp=None, frame_id=None, seq=None):
    '''
    Create a sensor_msgs.PointCloud2 from an array
    of points.
    '''
    from sensor_msgs.msg import PointCloud2, PointField

    msg = PointCloud2()
    # assert(points.shape == colors.shape)
    colors = np.zeros(points.shape)
    buf = []

    if stamp:
        msg.header.stamp = stamp
    if frame_id:
        msg.header.frame_id = frame_id
    if seq:
        msg.header.seq = seq
    if len(points.shape) == 3:
        msg.height = points.shape[1]
        msg.width = points.shape[0]
    else:
        N = len(points)
        xyzrgb = np.array(np.hstack([points, colors]), dtype=np.float32)
        msg.height = 1
        msg.width = N

    msg.fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1),
        PointField('r', 12, PointField.FLOAT32, 1),
        PointField('g', 16, PointField.FLOAT32, 1),
        PointField('b', 20, PointField.FLOAT32, 1)
    ]
    msg.is_bigendian = False
    msg.point_step = 24
    msg.row_step = msg.point_step * N
    msg.is_dense = True
    msg.data = xyzrgb.tostring()

    return msg


def do_icp(cloud_in, 
           camera_pose_matrix, 
           pub_filtered_cloud, 
           mesh_cloud, 
           pub_pose_cloud, 
           pose_estimate):

    
    # loc_scale = np.array([0.45, 1.5, 0.73])
    # ori = [-1, 0, 0, 1]
    loc_scale = pose_estimate['location']
    ori = pose_estimate['quaternion']
    ori_euler = tf.transformations.euler_from_quaternion(pose_estimate['quaternion'])

    R = tf.transformations.quaternion_matrix(ori)
    T = tf.transformations.translation_matrix(loc_scale)

    total_transform = tf.transformations.concatenate_matrices(T, R)
    cloud_filtered_array = transform_cloud(mesh_cloud, mat=np.copy(total_transform))
    cloud_color = np.zeros(cloud_filtered_array.shape[0])
    # mesh_cloud_cp = np.copy(mesh_cloud)
    # ros_msg = xyzrgb_array_to_pointcloud2(
    #     cloud_filtered_array, cloud_color, rospy.Time.now(), "/base_footprint"
    # )
    # pub_pose_cloud.publish(ros_msg)
    # print(total_transform)
    # rospy.logwarn("Num points after downsample and filter : {}".format(cloud_filtered_array.shape[0]))
    
    cloud_pose = pcl.PointCloud()
    cloud_pose.from_array(cloud_filtered_array)
    # print(cloud_filtered_array)
    print("Doing ICP")
    icp = cloud_pose.make_GeneralizedIterativeClosestPoint()
    converged, transf, estimate, fitness = icp.gicp(cloud_pose, cloud_in, max_iter=10)

    # icp = cloud_pose.make_IterativeClosestPoint()
    # converged, transf, estimate, fitness = icp.icp(cloud_pose, cloud_in)

    print('has converged:' + str(converged) + ' score: ' + str(fitness))
    print(str(transf))
    total_transform_icp = tf.transformations.concatenate_matrices(transf, total_transform)
    print(str(total_transform_icp))

    pose_output = {}
    pose_output['quaternion'] = tf.transformations.quaternion_from_matrix(total_transform_icp)
    pose_output['location'] = tf.transformations.translation_from_matrix(total_transform_icp)
    pose_output['matrix'] = total_transform_icp

    # Keep only x,y, and yaw from ICP result
    yaw_icp   = math.atan2(transf[1, 0], transf[0, 0])
    yaw_in    = ori_euler[2]
    cos_term  = math.cos(yaw_in + yaw_icp)
    sin_term  = math.sin(yaw_in + yaw_icp)
    total_yaw = math.atan2(sin_term, cos_term)
    pose_output['quaternion'] = tf.transformations.quaternion_from_euler(ori_euler[0], ori_euler[1], total_yaw)
    pose_output['location'][2] = loc_scale[2]
    R = tf.transformations.quaternion_matrix(pose_output['quaternion'])
    T = tf.transformations.translation_matrix(pose_output['location'])

    total_transform_icp = tf.transformations.concatenate_matrices(T, R)


    total_transform_icp_cam = \
        tf.transformations.concatenate_matrices(camera_pose_matrix, total_transform_icp)
    print(total_transform_icp_cam)
    pose_output_cam = {}
    pose_output_cam['quaternion'] = tf.transformations.quaternion_from_matrix(total_transform_icp_cam)
    pose_output_cam['location'] = tf.transformations.translation_from_matrix(total_transform_icp_cam)
    pose_output_cam['matrix'] = total_transform_icp_cam


    cloud_filtered_array = transform_cloud(
        mesh_cloud, mat=total_transform_icp)
    cloud_color = np.zeros(cloud_filtered_array.shape[0])
    ros_msg = xyzrgb_array_to_pointcloud2(
        cloud_filtered_array, cloud_color, rospy.Time.now(), "/base_footprint"
    )
    pub_pose_cloud.publish(ros_msg)

    return pose_output, pose_output_cam

def image_to_cloud(depthData, depthScale, camera_intrinsic_matrix):
    '''
        Convert image to point cloud in camera frame
    '''
    depthHeight = depthData.shape[0]
    depthWidth =  depthData.shape[1]
    aligned = np.zeros((depthHeight, depthWidth, 3))
    point_cloud = []
    fx_d = camera_intrinsic_matrix[0,0]
    fy_d = camera_intrinsic_matrix[1,1]
    cx_d = camera_intrinsic_matrix[0,2]
    cy_d = camera_intrinsic_matrix[1,2]

    for v in range(1, (depthHeight)):
        for u in range(1, (depthWidth)):
            #  Apply depth intrinsics
            z = depthData[v,u]/depthScale
            x = ((u - cx_d) * z) / fx_d
            y = ((v - cy_d) * z) / fy_d
            
            if z <= 0:
                continue
            #  Apply the extrinsics
            # transformed = np.transpose(np.matmul(extrinsics, np.array([x, y, z, 1])))
            # aligned[v,u,1] = transformed[1]
            # aligned[v,u,2] = transformed[2]
            # aligned[v,u,3] = transformed[3]
            aligned[v,u,0] = x
            aligned[v,u,1] = y
            aligned[v,u,2] = z
            point_cloud.append([x, y, z])

    point_cloud = np.array(point_cloud, dtype=np.float32)
    return point_cloud

def cloud_to_image(depthHeight, depthWidth, depthScale, point_cloud, camera_intrinsic_matrix):
    '''
        Convert a point cloud in camera frame to an image
        @point_cloud : np.array with x,y,z in columns
    '''
    output_img = np.zeros((depthHeight, depthWidth))
    fx_rgb = camera_intrinsic_matrix[0,0]
    fy_rgb = camera_intrinsic_matrix[1,1]
    cx_rgb = camera_intrinsic_matrix[0,2]
    cy_rgb = camera_intrinsic_matrix[1,2]
    # import pdb
    # pdb.set_trace()
    rmin = np.inf
    cmin = np.inf
    rmax = -np.inf
    cmax = -np.inf

    for u in range(0, point_cloud.shape[0]):
        #  Apply RGB intrinsics
        x = (point_cloud[u,0] * fx_rgb / point_cloud[u,2]) + cx_rgb
        y = (point_cloud[u,1] * fy_rgb / point_cloud[u,2]) + cy_rgb
        
        # print("x:{}, y:{}, depth : {}".format(x,y,point_cloud[u,2]))
        # "x" and "y" are indices into the RGB frame, but they may contain
        # invalid values (which correspond to the parts of the scene not visible
        # to the RGB camera.
        # Do we have a valid index?
        if x > depthWidth or y > depthHeight or x < 0 or y < 0 or np.isnan(x) or np.isnan(y):
            continue
        
        #  Need some kind of interpolation. I just did it the lazy way
        x = int(round(x))
        y = int(round(y))
        rmin = min(rmin, x)
        rmax = max(rmax, x)
        cmin = min(cmin, y)
        cmax = max(cmax, y)

        output_img[y,x] = point_cloud[u,2] * depthScale
        # aligned(v,u,4) = single(rgbData(y, x, 1);
        # aligned(v,u,5) = single(rgbData(y, x, 2);
        # aligned(v,u,6) = single(rgbData(y, x, 3);
        
    # Clockwise rectangle corners
    contours = np.array([[rmin, cmin], [rmin, cmax], [rmax, cmax], [rmax, cmin]])
    cv2.fillPoly(output_img, pts=[contours], color=(1))    
    return output_img

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

    for v in range(0, (depthHeight)):
        for u in range(0, (depthWidth)):
            #  Apply depth intrinsics
            z = depthData[v,u]/depthScale
            x = ((u - cx_d) * z) / fx_d
            y = ((v - cy_d) * z) / fy_d
            
            if z <= 0:
                continue
            #  Apply the extrinsics
            # transformed = np.transpose(np.matmul(extrinsics, np.array([x, y, z, 1])))
            # aligned[v,u,1] = transformed[1]
            # aligned[v,u,2] = transformed[2]
            # aligned[v,u,3] = transformed[3]
            aligned[v,u,0] = x
            aligned[v,u,1] = y
            aligned[v,u,2] = z

    for v in range(0, (depthHeight)):
        for u in range(0, (depthWidth)):
            #  Apply RGB intrinsics
            x = (aligned[v,u,0] * fx_rgb / aligned[v,u,2]) + cx_rgb
            y = (aligned[v,u,1] * fy_rgb / aligned[v,u,2]) + cy_rgb
            
            # "x" and "y" are indices into the RGB frame, but they may contain
            # invalid values (which correspond to the parts of the scene not visible
            # to the RGB camera.
            # Do we have a valid index?
            if x > depthWidth or y > depthHeight or x < 0 or y < 0 or np.isnan(x) or np.isnan(y):
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
        
