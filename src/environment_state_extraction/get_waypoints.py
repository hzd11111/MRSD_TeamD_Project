import random
import time
import numpy as np
import cv2
import carla
import glob
import os
import sys








actor_list = []
try:
	client = carla.Client('localhost', 2000)
	client.set_timeout(2.0)
	print(client.get_available_maps())
	
	world = client.get_world()
	#world = client.load_world('Racetrack02')
	#world = client.load_world('NewTrack')

	blueprint_library = world.get_blueprint_library()
	
	bp = blueprint_library.filter('model3')[0]
	print(bp)
	
	#spawn_point = carla.Transform(carla.Location(x=46.16, y=73.51, z=20))#random.choice(world.get_map().get_spawn_points())
	spawn_point = random.choice(world.get_map().get_spawn_points())
	
	vehicle = world.spawn_actor(bp, spawn_point)
	
	vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0))
	
	actor_list.append(vehicle)

	carla_map = world.get_map()
	#all_waypoints = carla_map.generate_waypoints(distance=1.0)
	
	closest_waypoint = carla_map.get_waypoint(vehicle.get_location(), project_to_road=True)
	print(closest_waypoint.section_id)
	

	waypoints = carla_map.generate_waypoints(distance=1.0)

	waypoint_tuple_list = carla_map.get_topology()
	for w in waypoints:
		#if(w.section_id==2):
		if(w.road_id == 12):
			world.debug.draw_string(w.transform.location, 'O', draw_shadow=False,
	                                   color=carla.Color(r=255, g=0, b=0), life_time=120.0,
	                                   persistent_lines=True)

	#location = vehicle.get_location()
	#nearest_waypoint = carla_map.get_waypoint(location, project_to_road=True)
	#waypoint_on_map = carla.get_nearest_waypoint_same_lane(nearest_waypoint, all_waypoints)  

	#left_lane_waypoint = nearest_waypoint.get_left_lane()
	#right_lane_waypoint = nearest_waypoint.get_right_lane()
	#print(location, nearest_waypoint, left_lane_waypoint, right_lane_waypoint)

	#print(all_waypoints)
	# while(True):
	# 	print(vehicle.get_location())
	# 	print(vehicle.get_velocity())
	# Getting the RGB and Depth Sensor

	# bluepirint_camera = blueprint_library.find('sensor.camera.rgb')
	# bluepirint_camera.set_attribute('image_size_x', f'{im_width}')
	# bluepirint_camera.set_attribute('image_size_y', f'{im_height}')
	# bluepirint_camera.set_attribute('fov', '110')

	# spawn_point_camera = carla.Transform(carla.Location(x=2.5, z=0.7))
	# camera = world.spawn_actor(bluepirint_camera, spawn_point_camera, attach_to=vehicle)

	# actor_list.append(camera)
	# camera.listen(lambda data: process_image(data))

	time.sleep(20)
	
	
	
finally:
	
	print("Destroying Actors")
	for actor in actor_list:
		actor.destroy()
	print('done!')
