#!/usr/bin/env python
from __future__ import print_function
import roslib
import sys
import math
import rospy
from geometry_msgs.msg import Twist

class Bebop2(object):
	"""docstring for [object Object]."""
	def __init__(self,
				 forward_speed,
				 yaw_cmd_value,
				 angular_speed,
				 vel_cmd_pub,
		 		 dubin_omega,
				 ):
		self.forward_speed = forward_speed
		self.yaw_cmd_value = math.pi/6
		self.angular_speed = angular_speed
		self.vel_cmd_pub = vel_cmd_pub
		self.depth_compensation = forward_speed * 0.9 / angular_speed
		self.dubin_omega = 1
		self.kill_thread = False
		self.done = False

	def goForward(self):
		vel_msg = Twist()
		vel_msg.linear.x = self.forward_speed
		vel_msg.linear.y = 0
		vel_msg.linear.z = 0
		vel_msg.angular.x = 0
		vel_msg.angular.y = 0
		vel_msg.angular.z = 0
		self.vel_cmd_pub.publish(vel_msg)

	def moveDrone(self, waypoint, depth):
		direction = 1 if waypoint > 0 else -1
		vel_msg = Twist()
		vel_msg.linear.x = self.forward_speed
		vel_msg.linear.y = 0
		vel_msg.linear.z = 0
		vel_msg.angular.x = 0
		vel_msg.angular.y = 0
		vel_msg.angular.z = self.yaw_cmd_value * direction

		print('Velocities: x =', vel_msg.linear.x, 'yaw =', vel_msg.angular.z)

		theta = math.atan(waypoint / depth)
		time_of_rotation = abs(theta) / self.angular_speed
		print('Time of rotation:', time_of_rotation)
		start_time = rospy.Time.now().to_sec()
		while( rospy.Time.now().to_sec() - start_time < time_of_rotation ):
			self.vel_cmd_pub.publish(vel_msg)

		time_of_flight = (math.sqrt(depth**2 + waypoint**2)- self.depth_compensation) / self.forward_speed
		print('Time of flight:', time_of_flight)
		start_time = rospy.Time.now().to_sec()
		while( rospy.Time.now().to_sec() - start_time < time_of_flight ):
			self.goForward()
		start_time = rospy.Time.now().to_sec()
		vel_msg.angular.z = -vel_msg.angular.z
		while( rospy.Time.now().to_sec() - start_time < time_of_rotation ):
			self.vel_cmd_pub.publish(vel_msg)
		print("Inside Wile loop")


	def dubinsMoveDrone(self, mode, pathlength):
		print(mode)
		print(pathlength)
		vel_msg = Twist()
		vel_msg.linear.x = self.forward_speed
		vel_msg.linear.y = 0
		vel_msg.linear.z = 0
		vel_msg.angular.x = 0
		vel_msg.angular.y = 0	
		vel_msg.angular.z = 0
		time = [1,1,1];
		time[0] = pathlength[0]/self.dubin_omega;
		time[1] = pathlength[1]/(self.forward_speed*1)
		time[2] = pathlength[2]/self.dubin_omega;
		i = 0
		start_time = rospy.Time.now().to_sec()
		# while( rospy.Time.now().to_sec() - start_time < 10 and not self.kill_thread):
		# 	self.vel_cmd_pub.publish(vel_msg)
		for modes in mode:
			if modes == 'R':
				direction = -1
			elif modes == 'S': 
				direction = 0
			else:
				direction = 1

			vel_msg.angular.z = self.dubin_omega* direction
			
			start_time = rospy.Time.now().to_sec()
			print(self.kill_thread)
			while( rospy.Time.now().to_sec() - start_time < time[i] and not self.kill_thread and time[i] > 0.05):
				self.vel_cmd_pub.publish(vel_msg)
			i = i + 1

		self.done = True