#!/usr/bin/env python

""" Some useful functions. """

__author__ = "Mayank Singal"
__maintainer__ = "Mayank Singal"
__email__ = "mayanksi@andrew.cmu.edu"
__version__ = "0.1"

import carla
import numpy as np


class DataCollector:
    
    def __init__(self):
        
        self.min_dist_actor = 100000
        self.max_dist_lane = 10000
        
        self.start_position = None#[10000,10000]
        self.curr_distance_travelled = 0
        
        self.avg_min_dist_actor = 0
        self.avg_max_dist_lane = 0
        self.avg_distance_travelled = 0
        
        self.counts = -1
        
    def update(self, actor_dist, lane_dist, curr_position, ignore_lane=False):
        
        if(self.start_position is None):
            self.start_position = curr_position
        else:
                
            self.min_dist_actor = min(self.min_dist_actor, actor_dist)
            if(not ignore_lane):
                self.max_dist_lane = min(self.max_dist_lane, lane_dist)
            
            self.curr_distance_travelled = np.sqrt((curr_position[0] - self.start_position[0])**2 + (curr_position[1] - self.start_position[1])**2)
            
    def reset(self):
        self.counts += 1
        
        if(self.counts != 0 and self.min_dist_actor < 1000):
                
            self.avg_min_dist_actor = (self.avg_min_dist_actor * (self.counts-1) + self.min_dist_actor)/float(self.counts)
            self.avg_max_dist_lane = (self.avg_max_dist_lane * (self.counts-1) + self.max_dist_lane)/float(self.counts)
            self.avg_distance_travelled = (self.avg_distance_travelled * (self.counts-1) + self.curr_distance_travelled)/float(self.counts)
            
        self.min_dist_actor = 100000
        self.max_dist_lane = 10000
        self.start_position = None#[10000,10000]
         

    



def get_matrix(transform):

    """
    Creates matrix from carla transform.
    """

    rotation = transform.rotation
    location = transform.location
    c_y = np.cos(np.radians(rotation.yaw))
    s_y = np.sin(np.radians(rotation.yaw))
    c_r = np.cos(np.radians(rotation.roll))
    s_r = np.sin(np.radians(rotation.roll))
    c_p = np.cos(np.radians(rotation.pitch))
    s_p = np.sin(np.radians(rotation.pitch))
    matrix = np.matrix(np.identity(4))
    matrix[0, 3] = location.x
    matrix[1, 3] = location.y
    matrix[2, 3] = location.z
    matrix[0, 0] = c_p * c_y
    matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
    matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
    matrix[1, 0] = s_y * c_p
    matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
    matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
    matrix[2, 0] = s_p
    matrix[2, 1] = -c_p * s_r
    matrix[2, 2] = c_p * c_r
    return matrix



def create_bb_points(vehicle) -> np.ndarray:
    """
    Returns 3D bounding box for a vehicle.
    """

    cords = np.zeros((8, 4))
    extent = vehicle.bounding_box.extent
    cords[0, :] = np.array([extent.x, extent.y, -extent.z, 1])
    cords[1, :] = np.array([-extent.x, extent.y, -extent.z, 1])
    cords[2, :] = np.array([-extent.x, -extent.y, -extent.z, 1])
    cords[3, :] = np.array([extent.x, -extent.y, -extent.z, 1])
    cords[4, :] = np.array([extent.x, extent.y, extent.z, 1])
    cords[5, :] = np.array([-extent.x, extent.y, extent.z, 1])
    cords[6, :] = np.array([-extent.x, -extent.y, extent.z, 1])
    cords[7, :] = np.array([extent.x, -extent.y, extent.z, 1])
    return cords



def project(bb, normal):
    projected_points = []
    for point in bb:
        dist = normal[0]*point[0]+normal[1]*point[1]+normal[2]
        alpha = np.arctan2(-normal[0],normal[1]) + np.pi/2
        xp,yp=point
        xp = point[0]-abs(dist)*np.cos(alpha)
        yp = point[1]-abs(dist)*np.sin(alpha)
        if round(abs(normal[0]*xp+normal[1]*yp+normal[2]),3)>0.01:   
            xp = point[0]+abs(dist)*np.cos(alpha)
            yp = point[1]+abs(dist)*np.sin(alpha)
        projected_points.append((xp,yp))
    if normal[0]==0:
        projected_points = sorted(projected_points, key=lambda b:b[0])
    else:
        projected_points = sorted(projected_points, key=lambda b:b[1])
    return np.array(projected_points[0]),np.array(projected_points[-1])

def project_and_calculate(bb1, normals1, bb2, normals2):
    # get max seperation between two objects (max seperation is closest meaningful seperation)
    closest = 0
    normals = normals1+normals2
    for normal in normals:
        l1,r1 = project(bb1,normal)
        l2,r2 = project(bb2,normal)
        if normal[0]!=0:
            if l2[1] > r1[1] and r2[1] > r1[1]:
                # no intersection
                dist = np.linalg.norm(l2-r1)
            elif l1[1] > r2[1] and r1[1] > r2[1]:
                # no intersecton
                dist = np.linalg.norm(l1-r2)
            else:
                dist=0
            if dist > closest:
                closest=dist
        if normal[0]==0:
            if l2[0] > r1[0] and r2[0] > r1[0]:
                # no intersection
                dist = np.linalg.norm(l2-r1)
            elif l1[0] > r2[0] and r1[0] > r2[0]:
                # no intersecton
                dist = np.linalg.norm(l1-r2)
            else:
                dist=0
            if dist > closest:
                closest=dist
    return closest
    
def get_normals(bb):
    normals = []
    for i in range(0,4):
        if i<2:
            x1,y1 = bb[i%4]
            x2,y2 = bb[(i+1)%4]
        else:
            x2,y2 = bb[i%4]
            x1,y1 = bb[(i+1)%4]
        a = (y1-y2)
        b = (x2-x1)
        c = x1*y2-x2*y1
        norm = np.linalg.norm([a,b])
        a, b, c = a/norm, b/norm, c/norm
        normals.append([a,b,c])
    return normals

def get_closest_distance(bb1,bb2):
    normals1 = get_normals(bb1)
    normals2 = get_normals(bb2)
    distance = project_and_calculate(bb1, normals1, bb2, normals2)
    return distance

def get_bounding_box(vehicle):

    '''
    d-a
    | |
    c-b        
    '''

    transform = vehicle.get_transform()
    coords = np.matmul(get_matrix(transform), create_bb_points(vehicle).T).T
    tmp_location = coords[0][:3]
    a = carla.Location()
    a.x = tmp_location[0,0]
    a.y = tmp_location[0,1]
    a.z = tmp_location[0,2]

    tmp_location = coords[1][:3]

    b = carla.Location()
    b.x = tmp_location[0,0]
    b.y = tmp_location[0,1]
    b.z = tmp_location[0,2]

    tmp_location = coords[2][:3]

    c = carla.Location()
    c.x = tmp_location[0,0]
    c.y = tmp_location[0,1]
    c.z = tmp_location[0,2]

    tmp_location = coords[3][:3]

    d = carla.Location()
    d.x = tmp_location[0,0]
    d.y = tmp_location[0,1]
    d.z = tmp_location[0,2]
    
    
    return [[d.x,d.y],[a.x,a.y],[b.x,b.y],[c.x,c.y]]
    