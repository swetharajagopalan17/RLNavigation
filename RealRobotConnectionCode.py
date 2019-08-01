#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 10:45:45 2019

@author: sunny
"""
import os
import numpy as np
import scipy as sp
import message_filters
import rospy
import cv2
import keras
import tensorflow as tf
os.environ['KERAS_BACKEND'] = 'tensorflow'
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist, TwistStamped
from std_msgs.msg import Float64
from move_base_msgs.msg import MoveBaseActionResult
from cv_bridge import CvBridge
import pandas as pd
import random
from keras.models import Model, load_model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization, Concatenate
import matplotlib.pyplot as plt

model = load_model('/home/isera2/catkin_sunny/src/sunny_pioneer/models/modelgpufull5/batch0network')
graph = tf.get_default_graph()
model.summary()
mergedimage = np.zeros((240,320,4))
cmdvel = []
count=0

odomlocalvector = []
vel_msg = Twist()
cmdvel = []
odomlocalvector = []
mergedimage = np.zeros((240,320,4))
inputimage = []
inputvector = []



############################################################################################################################

"""def gotrgb(image_rgb):
    print("Got RGB")
    
def gotdepth(image_depth):
    print("Got Depth")
    
def gotodom(odometry):
    print("Got Odom")
    
def gotpath(local_plan):
    print("Got Path")
"""

def predictcmdvel(inputimage, inputvector):
    global graph
    with graph.as_default():
        prediction = model.predict([inputimage, inputvector])
    return prediction

def resultcallback(result):
	global goalarray, failcount, randcount, faillock
	vel_msg.linear.x = 0
	vel_msg.linear.y = 0
	vel_msg.linear.z = 0
	vel_msg.angular.x = 0
	vel_msg.angular.y = 0
	vel_msg.angular.z = 0
	if(result.status.status == 3 or result.status.status == 4):
          cmd_vel_pub.publish(vel_msg)
          print("\nStopped")

def callback(image_rgb, image_depth, odometry, local_plan):
    global inputimage, inputvector, graph, model, count
    inputimage = []
    inputvector = []
    odomlocalvector = []

	# Solve all of perception here...
    #vel_msg = Twist()
    #cmdvel = []
    #odomlocalvector = []
    #mergedimage = np.zeros((240,320,4))
    
    #rgb and depth images
    imagergb = bridge.imgmsg_to_cv2(image_rgb, desired_encoding="bgr8")
    imaged = bridge.imgmsg_to_cv2(image_depth, desired_encoding="passthrough")
    
    #downsample images
    imaged2 = imaged
    imaged2 = imaged2[::2,::2]
    imagergb = imagergb[::2,::2]
    
    #Remove NaN values from depthimages
    imaged2 = np.where(np.isnan(imaged2), np.ma.array(imaged2, 
               mask = np.isnan(imaged2)).mean(axis = 0), imaged2)
    for i in range(len(imaged2)):
        for j in range(len(imaged2[1])):
            if imaged2[i,j] == 0:
                imaged2[i,j] = np.amax(imaged2)
            
            imaged2[i,j] = imaged2[i,j]*25.951
            if imaged2[i,j] > 255:
                imaged2[i,j] = 255
    
    #depthimages.insert(count, imaged2[:,:])
    #rgbimages.insert(count, imagergb)
    mergedimage[:,:,0:3] = imagergb[:,:,:]
    mergedimage[:,:,3] = imaged2[:,:]
    
    odomlocalvector.insert(0, odometry.pose.pose.position.x-local_plan.poses[1].pose.position.x)
    odomlocalvector.insert(1, odometry.pose.pose.position.y-local_plan.poses[1].pose.position.y)
    odomlocalvector.insert(2, odometry.pose.pose.position.z-local_plan.poses[1].pose.position.z)
    odomlocalvector.insert(3, odometry.pose.pose.orientation.x-local_plan.poses[1].pose.orientation.x)
    odomlocalvector.insert(4, odometry.pose.pose.orientation.y-local_plan.poses[1].pose.orientation.y)
    odomlocalvector.insert(5, odometry.pose.pose.orientation.z-local_plan.poses[1].pose.orientation.z)
    odomlocalvector.insert(6, odometry.pose.pose.orientation.w-local_plan.poses[1].pose.orientation.w)
    vector = np.array(odomlocalvector)
    
    inputimage = np.expand_dims(mergedimage, axis=0)
    inputvector = np.expand_dims(vector, axis=1)
    inputvector = np.expand_dims(inputvector, axis=0)
    
    prediction = predictcmdvel(inputimage, inputvector)
   
    vel_msg.linear.x = 0
    vel_msg.linear.y = 0
    vel_msg.linear.z = 0
    vel_msg.angular.x = 0
    vel_msg.angular.y = 0
    vel_msg.angular.z = 0
    
    print("\nPredicted Linear:" + str(prediction[0,0]) + ", Angular: " + str(prediction[0,1]))
    #print("Correct Linear: " + str(cmdvel.twist.linear.x) + ", Angular: " + str(cmdvel.twist.angular.z) + "\n")

                                   
    cmd_vel_pub.publish(vel_msg)
    
    
############################################################################################################################

image_rgb_sub = message_filters.Subscriber('/camera/rgb/image_raw', Image)
image_depth_sub = message_filters.Subscriber('/camera/depth_registered/image_raw', Image)
odom_sub = message_filters.Subscriber('/RosAria/pose', Odometry)
local_goal_sub = message_filters.Subscriber('/move_base/EBandPlannerROS/global_plan', Path)
#cmd_vel_sub = message_filters.Subscriber('RosAria/cmd_vel_stamped', TwistStamped)
rospy.Subscriber("/move_base/result", MoveBaseActionResult, resultcallback)

#rospy.Subscriber('/camera/rgb/image_raw', Image, gotrgb)
#rospy.Subscriber('/camera/depth_registered/image_raw', Image, gotdepth)
#rospy.Subscriber('/RosAria/pose', Odometry, gotodom)
#rospy.Subscriber('/move_base/EBandPlannerROS/global_plan', Path, gotpath)

bridge = CvBridge()

cmd_vel_pub = rospy.Publisher('/RosAria/cmd_vel2', Twist, queue_size=1) 
rospy.init_node('cnn_navigation', anonymous=True)

ts = message_filters.ApproximateTimeSynchronizer([image_rgb_sub, image_depth_sub, odom_sub, local_goal_sub], 1, 0.8)
ts.registerCallback(callback)
rospy.spin()





#////////////////////////////////////////////////////////////////////////////
