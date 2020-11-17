#!/usr/bin/env python
# coding: utf-8

# ### Steps to record a Run

# 1. Run CARLA
# 2. RUN carla_node.py
# 3. Run this script when the system is up and running

# In[1]:


import random
import sys
import time
import carla
from carla import ColorConverter as cc
import numpy as np
import numpy
from PIL import Image
import cv2
from functools import partial
import argparse

HEIGHT = 720 #720p
WIDTH = 1280
QUALITY = 45
EGO_NAME = 'ego'


# camera sensor callback function, gets and image and saves it to the folder
def to_bgra_array(image):
    """Convert a CARLA raw image to a BGRA numpy array."""
    array = numpy.frombuffer(image.raw_data, dtype=numpy.dtype("uint8"))
    array = numpy.reshape(array, (image.height, image.width, 4))
    return array

def to_rgb_array(image):
    """Convert a CARLA raw image to a RGB numpy array."""
    array = to_bgra_array(image)
    # Convert BGRA to RGB.
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    return array

def process_video(writer, image):
    '''Process sensor data and save as video'''
    try:
        image = to_rgb_array(image)
        bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        writer.write(bgr_image)
    except KeyboardInterrupt:
        print("Stopped in process_video function")
        writer.release()

def get_camera(ego_vehicle, fn):
    camera_bp =  world.get_blueprint_library().find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', str(WIDTH))
    camera_bp.set_attribute('image_size_y', str(HEIGHT))

    transform = carla.Transform(carla.Location(x=-15.5, z=12.5), carla.Rotation(pitch=-20.))
    Attachment = carla.AttachmentType

    camera = world.spawn_actor(camera_bp, transform, attach_to=ego_vehicle, attachment_type=Attachment.Rigid)
    camera.listen(fn)
    return camera

def get_ego_vehicle():
    world.wait_for_tick(seconds=10)
    ego_vehicle = None
    
    while ego_vehicle == None:
        all_vehicles = world.get_actors().filter('vehicle.*')
        for vehicle in all_vehicles:
            if vehicle.attributes['role_name'] == EGO_NAME:
                ego_vehicle = vehicle
                break
                
        if ego_vehicle is not None: break
            
    return ego_vehicle
parser = argparse.ArgumentParser(description='Save Carla Video.')
parser.add_argument('-f', '--filename', help='filename for video', default="output.mp4")
args = parser.parse_args()

fourcc = cv2.VideoWriter_fourcc(*'MP4V')
writer = cv2.VideoWriter(args.filename, fourcc, 45.0, (WIDTH, HEIGHT))
process_video_wrapped = partial(process_video, writer)
client = carla.Client("localhost", 2000)
client.set_timeout(10.0)
world = client.get_world()
Map = world.get_map()
world.wait_for_tick(seconds=60)

# all_vehicles = world.get_actors().filter('vehicle.*')
# for vehicle in all_vehicles:
#     if vehicle.attributes['role_name'] == EGO_NAME:
#         ego_vehicle = vehicle
#         break
# print(ego_vehicle)

# camera_bp =  world.get_blueprint_library().find('sensor.camera.rgb')
# camera_bp.set_attribute('image_size_x', str(WIDTH))
# camera_bp.set_attribute('image_size_y', str(HEIGHT))

# transform = carla.Transform(carla.Location(x=-15.5, z=12.5), carla.Rotation(pitch=8.0))
# Attachment = carla.AttachmentType

# camera = world.spawn_actor(camera_bp, transform, attach_to=ego_vehicle, attachment_type=Attachment.SpringArm)
# camera.listen(process_video_wrapped)

while True: #loops over multiple runs
    ego_vehicle = get_ego_vehicle()
    camera = get_camera(ego_vehicle, process_video_wrapped)
    
    try:
        while True:
            world.wait_for_tick(seconds=60)
            
            # ego id may change between runs, so need to reattach camera
            if ego_vehicle.get_location().x == 0.000000:
                camera.destroy()
                ego_vehicle = get_ego_vehicle()
                print(ego_vehicle)
                camera = get_camera(ego_vehicle, process_video_wrapped)

    except KeyboardInterrupt:
        camera.destroy()
        if writer.isOpened():
            writer.release()
        print('Exit')
    finally:
        camera.destroy()
        if writer.isOpened():
            writer.release()
        break
