#!/usr/bin/env python
import rospy 
import tf
import tf2_ros
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Pose
from geometry_msgs.msg import TransformStamped
from bebop_vel_ctrl.msg import Debug
from math import sin,cos

x = 0
y = 0
z = 0

vel_x = 0
vel_y = 0

yaw   = 0
pitch = 0
roll  = 0

omega = 0

br = tf2_ros.TransformBroadcaster()
t = TransformStamped()

odom = Odometry()

prev = 0

def callback(msg):
	global vel_x 
	global vel_y 
	global z 
	global yaw   
	global pitch 
	global roll  
	global omega 
	global prev
	global x
	global y

	vel_x = msg.beb_vx_m
	vel_y = msg.beb_vy_m
	z     = msg.beb_alt_m
	
	yaw   = msg.beb_yaw_rad - yaw_init
	pitch = msg.beb_pitch_rad
	roll  = msg.beb_roll_rad

	omega = msg.beb_vyaw_rad

	cur  = rospy.get_time()
	dt = cur - prev 
	prev = rospy.get_time()
	del_x = (vel_x*cos(yaw) - vel_y*sin(yaw))*dt
	del_y = (vel_x*sin(yaw) - vel_y*cos(yaw))*dt
	x += del_x
	y += del_y
	
	print yaw

	quaternion = tf.transformations.quaternion_from_euler(roll, pitch, yaw)

	t.header.stamp = rospy.Time.now()
	t.header.frame_id = "estimated_odom"
	t.child_frame_id = "base_link"
	t.transform.translation.x = x
	t.transform.translation.y = y
	t.transform.translation.z = z
	t.transform.rotation.x = quaternion[0]
	t.transform.rotation.y = quaternion[1]
	t.transform.rotation.z = quaternion[2]
	t.transform.rotation.w = quaternion[3]

	br.sendTransform(t)

	pose = Pose()
	pose.position.x = x
	pose.position.y = y
	pose.position.z = z
	pose.orientation.x = quaternion[0]
	pose.orientation.y = quaternion[1]
	pose.orientation.z = quaternion[2]
	pose.orientation.w = quaternion[3]

	twist = Twist()
	twist.linear.x  = vel_x
	twist.linear.y  = vel_y
	twist.angular.z = omega 

	odom.header.stamp = rospy.Time.now()
	odom.header.frame_id = "/odom"
	odom.pose.pose = pose
	odom.twist.twist = twist

	pub.publish(odom)

pub = rospy.Publisher('/estimated_odom', Odometry, queue_size = 10)

rospy.init_node('odometry', anonymous = True)

first_msg = rospy.wait_for_message("/vel_ctrl/debug", Debug)

yaw_init = first_msg.beb_yaw_rad

rospy.Subscriber("/vel_ctrl/debug", Debug, callback)

rospy.spin()