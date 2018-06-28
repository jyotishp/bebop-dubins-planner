#!/usr/bin/env python

# Load system modules
from __future__ import print_function
import sys, math, threading, time

# Load ROS modules
import roslib, rospy
roslib.load_manifest('navigator')
from std_msgs.msg import String, Empty
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from apriltags_ros.msg import AprilTagDetection, AprilTagDetectionArray

# Load custom modules
from planner import DubinsPlanner as Planner
from drone import Bebop2 as Drone

# Main function
def main(args):
	# Intiate the ROS node
	rospy.init_node('navigator', anonymous=True)
	
	# Drone control topics
	vel_cmd_pub = rospy.Publisher('/vservo/cmd_vel', Twist, queue_size=3)
	land = rospy.Publisher('/bebop/land', Empty, queue_size=3)
	
	# Load drone model
	drone = Drone(
					forward_speed = 1,
					yaw_cmd_value = 1,
					angular_speed = 1,
					vel_cmd_pub = vel_cmd_pub,
					dubin_omega = 1)

	# Planner parameters (All in meters)
	planner = Planner(
		min_safe_distance = 5,
		window_size = (0.5, 0.5), # (x, y)
		grid_size = (10, 10), # (x, y)
		obstacle_radius = 0.5,
		goal = (20, 0, 0), # (x, y, heading)
		next_goal = [(15,5,math.pi/2),(20,5,0),(30,0,0)],
		error_margin = 0.1,
		drone = drone,
		land = land
		)

	# Subscribe to required topics
	tag_sub = rospy.Subscriber('/tag_detections', AprilTagDetectionArray, planner.detectAprilTags)
	odometry_sub = rospy.Subscriber('/bebop/odom', Odometry, planner.updateOdometry)

	# Thread to run the planner
	print('Starting planner in 5 seconds...')
	time.sleep(5)
	planner_thread = threading.Thread(target = planner.run)
	planner_thread.start()

	try:
		rospy.spin()
	except Exception as e:
		print('Shutting Down!')

if __name__ == '__main__':
	main(sys.argv)
