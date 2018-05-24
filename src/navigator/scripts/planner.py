#!/usr/bin/env python

# Import system modules
from __future__ import print_function
import math, time, threading

# Import python modules
import numpy as np
from matplotlib import pyplot as plt
import cv2

# Import ROS modeules
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError

# Import custom modules
from dubins import dubins_path_planning

class DubinsPlanner():
	"""
		- Generate a grid with all detected tags on it.
		- Detect free space on the grid and get waypoint in x wrt to drone.
		- If waypoint is 0 continue.
		- Else replan.
	"""
	def __init__(self, 
				min_safe_distance,
				window_size,
				grid_size,
				obstacle_radius,
				goal,
				error_margin,
				drone
				):
		self.min_safe_distance = min_safe_distance
		self.goal = goal
		self.error_margin = error_margin
		self.drone = drone
		self.grid = Grid(grid_size, obstacle_radius)
		self.control_point = 0
		self.next_waypoint = None
		self.move = False
		self.initialized = False
		self.kill_thread = False

	def resetSearch(self):
		self.control_point = 0

	# Callback for odometry
	def updateOdomentry(self, odometry):
		# I have no idea how odometry msg looks like
		# Update the value of self.odometry
		# self.odometry = {x, y, heading}
		pass

	# Plan dubins curve and move drone along it
	def run(self):
		if not self.initialized and self.odometry:
			# Plan initial trajectory
			_, _, _, mode, _, pathlength = dubins_path_planning(
				sx = 0,
				sy = 0,
				syaw = 0,
				ex = goal[0],
				ey = goal[1],
				eyaw = goal[2],
				c = 1
				)
			self.move = True
			self.initialized = True
		while self.move:
			if goalReached():
				self.move = False
				self.kill_thread = True
			if self.next_waypoint:
				# Plan a dubins curve
				# Kill previous moveDrone thread
				# Start a new thread for moveDrone (use while self.kill)
				# Limit the loop rate
			# Replan to goal if waypoint reached
		print('Reached Goal!')

	# Check if the goal is reached
	def goalReached(self):
		margin_x = self.goal[0] * self.error_margin
		margin_y = self.goal[1] * self.error_margin
		if abs(self.goal[0] - self.odometry['x']) < margin_x 
		and abs(self.goal[1] - self.odometry['y']) < margin_y:
			return True

	# Callback for apriltags subscriber
	def detectAprilTags(self, tags):
		# Put all tags on a grid
		self.grid.pupulate(tags)
		# Detect free space and Set self.next_waypoint
		self.next_waypoint = self.getWaypoint()
		# Reset the grid
		self.grid.reset()

	# Helper function to calculate waypoint
	def getWaypoint(self):
		# Slide window and check for freespace
		freespace = False
		while not freespace and self.control_point < self.grid.size[0]:
			# Select the window region
			sub_space_left, sub_space_right = self.crop()
			# Check for obstacles
			obstacles_left = np.count_nonzero(sub_space_left)
			obstacles_right = np.count_nonzero(sub_space_right)
			# Move window if there are obstacles
			if obstacles_left > 0 and obstacles_right > 0:
				self.control_point += 1
			# Return None if window was not moved, else distance of waypoint in metres.
			else:
				self.control_point = 1 if obstacles_left > 0 else -1
				return None if self.control_point == 0 else self.control_point/10
		# Halt if no free space is detected
		self.move = False
		print('No free space detected. Halting!')

	# This is bad. Refactor!
	def crop(self):
		return self.grid.grid[self.grid.origin['y'] - round(self.window_size*10):self.grid.origin['y'] - round(self.window_size*10), self.grid.origin['x'] - self.control_point - round(self.window_size*10):self.grid.origin['x'] - self.control_point + round(self.window_size*10)], self.grid.grid[self.grid.origin['y'] - round(self.window_size*10):self.grid.origin['y'] - round(self.window_size*10), self.grid.origin['x'] + self.control_point - round(self.window_size*10):self.grid.origin['x'] + self.control_point + round(self.window_size*10)]


class Grid(object):
	def __init__(self, grid_size, obstacle_radius):
		self.obstacle_radius = obstacle_radius
		self.size = grid_size
		self.origin = {
			'x': round(grid_size[0]/2),
			'y': round(grid_size[1]/2)
		}
		self.grid = np.zeros(10 * grid_size)
	
	def reset(self):
		self.grid = np.zeros(self.size)

	def populate(self, tags):
		for tag in tags:
			x = round(tag.pose.pose.position.x*10) + self.origin['x']
			y = round(tag.pose.pose.position.y*10) + self.origin['y']
			try:
				self.grid[y-self.obstacle_radius:y+self.obstacle_radius, x-self.obstacle_radius:x+self.obstacle_radius] = 1
			except Exception as e:
				print('Obstacle omitted dues to padding.')
