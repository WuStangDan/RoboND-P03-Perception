#!/usr/bin/env python

# Import modules
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
import pickle
from sensor_stick.srv import GetNormals
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from visualization_msgs.msg import Marker
from sensor_stick.marker_tools import *
from sensor_stick.msg import DetectedObjectsArray
from sensor_stick.msg import DetectedObject
from sensor_stick.pcl_helper import *

import rospy
import tf
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
from std_msgs.msg import Int32
from std_msgs.msg import String
from pr2_robot.srv import *
from rospy_message_converter import message_converter
import yaml


# Helper function to get surface normals
def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster

# Helper function to create a yaml friendly dictionary from ROS messages
def make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose):
    yaml_dict = {}
    yaml_dict["test_scene_num"] = test_scene_num.data
    yaml_dict["arm_name"]  = arm_name.data
    yaml_dict["object_name"] = object_name.data
    yaml_dict["pick_pose"] = message_converter.convert_ros_message_to_dictionary(pick_pose)
    yaml_dict["place_pose"] = message_converter.convert_ros_message_to_dictionary(place_pose)
    return yaml_dict

# Helper function to output to yaml file
def send_to_yaml(yaml_filename, dict_list):
    data_dict = {"object_list": dict_list}
    with open(yaml_filename, 'w') as outfile:
        yaml.dump(data_dict, outfile, default_flow_style=False)




# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):

    # TODO: Convert ROS msg to PCL data
    cloud = ros_to_pcl(pcl_msg)
    # TODO: Voxel Grid Downsampling
    vox = cloud.make_voxel_grid_filter()
    # Choose voxel size.
    LEAF_SIZE = 0.005
    # Set the voxel size.
    vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)
    # Call the filter function to downsample cloud.
    cloud_filtered = vox.filter()
    

    # TODO: PassThrough Filter
    passthrough_z = cloud_filtered.make_passthrough_filter()
    # Assign axis and range.
    filter_axis = 'z'
    passthrough_z.set_filter_field_name(filter_axis)
    axis_min = 0.6
    axis_max = 1.1
    passthrough_z.set_filter_limits(axis_min, axis_max)
    # Use filter function to obtain new point cloud.
    cloud_filtered = passthrough_z.filter()
    # Filter in x direction also.
    passthrough_x = cloud_filtered.make_passthrough_filter()
    filter_axis = 'x'
    passthrough_x.set_filter_field_name(filter_axis)
    axis_min = 0.35
    axis_max = 2.0
    passthrough_x.set_filter_limits(axis_min, axis_max)
    # Use filter function to obtain new point cloud.
    cloud_filtered = passthrough_x.filter()


    # Statistical outlier removal.
    outlier_filter = cloud_filtered.make_statistical_outlier_filter()
    # Set the number of neighboring points to analyze.
    outlier_filter.set_mean_k(20)
    # Threshold scale factor.
    x = 0.25
    # Any point with mean distance larger than mean distance + x*stddev will be an outlier.
    outlier_filter.set_std_dev_mul_thresh(x)
    cloud_filtered = outlier_filter.filter()
    #test_pub.publish(pcl_to_ros(cloud_objects)) 

    # TODO: RANSAC Plane Segmentation
    seg = cloud_filtered.make_segmenter()
    # Set the model you wish to fit.
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)
    # Max distance for a point to be considered to fit the model.
    max_distance = 0.01
    seg.set_distance_threshold(max_distance)
    # Perform segmentation.
    inliers, coefficients = seg.segment()
    # TODO: Extract inliers and outliers
    cloud_table = cloud_filtered.extract(inliers, negative=False)
    cloud_objects = cloud_filtered.extract(inliers, negative=True)
      

    # TODO: Euclidean Clustering
    location_cloud = XYZRGB_to_XYZ(cloud_objects)
    tree = location_cloud.make_kdtree()
    #Create cluster extraction.
    euclid_cluster = location_cloud.make_EuclideanClusterExtraction()
    # Set tolerances.
    euclid_cluster.set_ClusterTolerance(0.03)
    euclid_cluster.set_MinClusterSize(10)
    euclid_cluster.set_MaxClusterSize(2000)
    # Search k-d tree for clusters.
    euclid_cluster.set_SearchMethod(tree)
    # Extract indices for found clusters.
    cluster_indices = euclid_cluster.Extract()

    # TODO: Create Cluster-Mask Point Cloud to visualize each cluster separately
    cluster_color = get_color_list(len(cluster_indices))
    #print(len(cluster_indices))
    color_cluster_point_list = []

    for j, indices in enumerate(cluster_indices):
        for i, index in enumerate(indices):
            color_cluster_point_list.append([location_cloud[index][0],
                                             location_cloud[index][1],
                                             location_cloud[index][2],
                                             rgb_to_float(cluster_color[j])])
    # Create new cloud with clusters a unique color.
    cluster_cloud = pcl.PointCloud_PointXYZRGB()
    cluster_cloud.from_list(color_cluster_point_list)

    # TODO: Convert PCL data to ROS messages
    ros_cloud_table = pcl_to_ros(cloud_table)
    ros_cloud_objects = pcl_to_ros(cloud_objects)
    ros_cloud_cluster = pcl_to_ros(cluster_cloud)
    # TODO: Publish ROS messages
    pcl_objects_pub.publish(ros_cloud_objects)
    pcl_table_pub.publish(ros_cloud_table)
    pcl_cluster_pub.publish(ros_cloud_cluster)



    # Classify the clusters! (loop through each detected cluster one at a time)
    detected_objects_labels = []
    detected_objects = []

    for index, pts_list in enumerate(cluster_indices):
        # Grab the points for the cluster
        pcl_cluster = cloud_objects.extract(pts_list)
        # Compute the associated feature vector
        ros_cluster = pcl_to_ros(pcl_cluster)
        color_hist = compute_color_histograms(ros_cluster, using_hsv=True)
        normals = get_normals(ros_cluster)
        normal_hist = compute_normal_histograms(normals)
        features = np.concatenate((color_hist, normal_hist))

        # Make the prediction
        prediction = clf.predict(scaler.transform(features.reshape(1,-1)))
        label = encoder.inverse_transform(prediction)[0]
        detected_objects_labels.append(label)

        # Publish a label into RViz
        label_pos = list(location_cloud[pts_list[0]])
        label_pos[2] += 0.2
        object_markers_pub.publish(make_label(label, label_pos, index))

        # Add the detected object to the list of detected objects.
        do = DetectedObject()
        do.label = label
        do.cloud = ros_cluster
        detected_objects.append(do)
    # Publish the list of detected objects
    #rospy.loginfo('Detected {} objects: {}'.format(len(detected_objects_labels), detected_objects_labels))
    detected_objects_pub.publish(detected_objects)


    # Generate YAML file containing labels and centriods for everything in pick_list.
    labels = []
    centriods = []
    for item in pick_list:
        for i in range(len(detected_objects)):
            if detected_objects[i].label == item['name']:
                print("Found", item['name'])
                labels.append(item['name'])
                points_array = ros_to_pcl(detected_objects[i].cloud).to_array()
                cent = np.mean(points_array, axis=0)
                centriods.append((np.asscalar(cent[0]), np.asscalar(cent[1]), np.asscalar(cent[2])))

    yaml_list = []
    scene_num = Int32()
    object_name = String()
    arm = String()
    pick_pose = Pose()
    place_pose = Pose()

    scene_num.data = 2
    arm.data = 'none'
    for i in range(len(labels)):
        object_name.data = labels[i]
        pick_pose.position.x = centriods[i][0]
        pick_pose.position.y = centriods[i][1]
        pick_pose.position.z = centriods[i][2]
        yaml_dict = make_yaml_dict(scene_num, arm, object_name, pick_pose, place_pose)
        yaml_list.append(yaml_dict)

    send_to_yaml('output_none.yaml', yaml_list)

    #try:
    #    pr2_mover(detected_objects)
    #except rospy.ROSInterruptException:
    #    pass

# function to load parameters and request PickPlace service
def pr2_mover(object_list):

    # TODO: Initialize variables

    # TODO: Get/Read parameters

    # TODO: Parse parameters into individual variables

    # TODO: Rotate PR2 in place to capture side tables for the collision map

    # TODO: Loop through the pick list

        # TODO: Get the PointCloud for a given object and obtain it's centroid

        # TODO: Create 'place_pose' for the object

        # TODO: Assign the arm to be used for pick_place

        # TODO: Create a list of dictionaries (made with make_yaml_dict()) for later output to yaml format

        # Wait for 'pick_place_routine' service to come up
        rospy.wait_for_service('pick_place_routine')

        try:
            pick_place_routine = rospy.ServiceProxy('pick_place_routine', PickPlace)

            # TODO: Insert your message variables to be sent as a service request
            resp = pick_place_routine(TEST_SCENE_NUM, OBJECT_NAME, WHICH_ARM, PICK_POSE, PLACE_POSE)

            print ("Response: ",resp.success)

        except rospy.ServiceException, e:
            print "Service call failed: %s"%e

    # TODO: Output your request parameters into output yaml file



if __name__ == '__main__':


    # TODO: ROS node initialization
    rospy.init_node('clustering', anonymous=True)
    # TODO: Create Subscribers
    pcl_sub = rospy.Subscriber("/pr2/world/points", pc2.PointCloud2, pcl_callback, queue_size=1)
    # TODO: Create Publishers
    pcl_objects_pub = rospy.Publisher("/pcl_objects", PointCloud2, queue_size=1)
    pcl_table_pub = rospy.Publisher("/pcl_table", PointCloud2, queue_size=1)
    pcl_cluster_pub = rospy.Publisher("/pcl_cluster", PointCloud2, queue_size=1)
    detected_objects_pub = rospy.Publisher("/detected_objects", DetectedObjectsArray, queue_size=1)
    object_markers_pub = rospy.Publisher("/object_markers", Marker, queue_size=1)
    test_pub = rospy.Publisher("/test_pts", PointCloud2, queue_size=1)

    # TODO: Load Model From disk
    model = pickle.load(open('model.sav', 'rb'))
    clf = model['classifier']
    encoder = LabelEncoder()
    encoder.classes_ = model['classes']
    scaler = model['scaler']

    # Initialize color_list
    get_color_list.color_list = []

    # Load object list.
    pick_list = rospy.get_param('/object_list')

    # TODO: Spin while node is not shutdown
    while not rospy.is_shutdown():
        rospy.spin
