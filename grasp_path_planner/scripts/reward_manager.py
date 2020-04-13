import numpy as np
from abc import ABC, abstractmethod
from enum import Enum
import copy
# ROS Packages
from geometry_msgs.msg import Pose2D
from std_msgs.msg import String
from grasp_path_planner.msg import LanePoint
from grasp_path_planner.msg import Lane
from grasp_path_planner.msg import VehicleState
from grasp_path_planner.msg import RewardInfo
from grasp_path_planner.msg import EnvironmentState
from grasp_path_planner.msg import RLCommand
from grasp_path_planner.msg import Pedestrian
from grasp_path_planner.msg import PathPlan
from grasp_path_planner.srv import SimService, SimServiceResponse, SimServiceRequest
# other packages
from settings import *

# Convert to local
def convert_to_local(cur_vehicle, adj_vehicle):
        result_state = VehicleState()
        x = adj_vehicle.vehicle_location.x
        y = adj_vehicle.vehicle_location.y
        theta = adj_vehicle.vehicle_location.theta
        speed = adj_vehicle.vehicle_speed
        vx = speed*np.cos(theta)
        vy = speed*np.sin(theta)
        # get current_vehicle_speeds
        cvx = cur_vehicle.vehicle_speed*np.cos(cur_vehicle.vehicle_location.theta)
        cvy = cur_vehicle.vehicle_speed*np.sin(cur_vehicle.vehicle_location.theta)
        # make homogeneous transform
        H_Rot = np.eye(3)
        H_Rot[-1,-1] = 1
        H_Rot[0,-1] = 0
        H_Rot[1,-1] = 0
        H_Rot[0,0] = np.cos(cur_vehicle.vehicle_location.theta)
        H_Rot[0,1] = -np.sin(cur_vehicle.vehicle_location.theta)
        H_Rot[1,0] = np.sin(cur_vehicle.vehicle_location.theta)
        H_Rot[1,1] = np.cos(cur_vehicle.vehicle_location.theta)
        H_trans = np.eye(3)
        H_trans[0,-1] = -cur_vehicle.vehicle_location.x
        H_trans[1,-1] = -cur_vehicle.vehicle_location.y
        H = np.matmul(H_Rot,H_trans)
        # calculate and set relative position
        res = np.matmul(H, np.array([x,y,1]).reshape(3,1))
        result_state.vehicle_location.x = res[0,0]
        result_state.vehicle_location.y = res[1,0]
        # calculate and set relative orientation
        result_state.vehicle_location.theta = theta-cur_vehicle.vehicle_location.theta
        # calculate and set relative speed
        res_vel = np.array([vx-cvx,vy-cvy])
        result_state.vehicle_speed = speed # np.linalg.norm(res_vel)
        # print("ADJ-----------------")
        # print(adj_vehicle)
        # print("CUR-----------------")
        # print(cur_vehicle)
        # print("RESULT--------------")
        # print(result_state)
        # time.sleep(5)
        return result_state

# Parent Reward Class
class Reward(ABC):
    '''
    Implements positional costs and basic required interfaces for all rewards
    '''
    def __init__(self):
        self.min_vel = 15
        self.max_vel = 20
        self.min_dist = 0.5
        self.k_vel = 0.01
        self.p_vel = 0.01
        self.k_pos = 1
        self.k_col = -10
        self.k_succ = 20
        self.closest_dist = 1e5
        self.cum_vel_err = 0
        self.max_reward = 20
    
    def project(self, bb, normal):
        projected_points = []
        for point in bb:
            dist = normal[0]*point[0]+normal[1]*point[1]+normal[2]
            alpha = np.arctan(-normal[1]/normal[0])+np.pi/2
            xp,yp=point
            xp = point[0]-abs(dist)*np.sin(alpha)
            yp = point[1]-abs(dist)*np.cos(alpha)
            if round(abs(normal[0]*xp+normal[1]*yp+normal[2]),3)>0.01:   
                xp = point[0]+abs(dist)*np.sin(alpha)
                yp = point[1]+abs(dist)*np.cos(alpha)
            projected_points.append((xp,yp))
        projected_points = sorted(projected_points, key=lambda b:b[1])
        return np.array(projected_points[0]),np.array(projected_points[-1])

    def project_and_calculate(self, bb1, normals1, bb2, normals2):
        # get max seperation between two objects (max seperation is closest meaningful seperation)
        closest = 0
        normals = normals1+normals2
        for normal in normals:
            l1,r1 = self.project(bb1,normal)
            l2,r2 = self.project(bb2,normal)
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
        return closest

    def get_closest_distance(self,desc):
        # make a list of bounding boxes of adjacent vehicles
        # NPCs = desc.adjacent_lane_vehicles+front_vehicle_state+back_vehicle_state
        NPCs = desc.adjacent_lane_vehicles
        NPC_bb = []
        NPC_normals = []
        for vehicle in NPCs:
            cx,cy, = vehicle.vehicle_location.x, vehicle.vehicle_location.y
            l,w = vehicle.length, vehicle.width
            NPC_bb.append([[cx+l/2,cy-w/2], [cx+l/2, cy+w/2], [cx-l/2, cy+w/2], [cx-l/2, cy-w/2]])
            normals=[]
            for i in range(0,4):
                if i<2:
                    x1,y1 = NPC_bb[-1][i%4]
                    x2,y2 = NPC_bb[-1][(i+1)%4]
                else:
                    x2,y2 = NPC_bb[-1][i%4]
                    x1,y1 = NPC_bb[-1][(i+1)%4]
                a = (y2-y1)
                b = -(x2-x1)
                c = (y1*(x2-x1)-(y2-y1)*x1)
                norm = np.linalg.norm([a,b])
                a, b, c = a/norm, b/norm, c/norm
                normals.append([a,b,c])
            NPC_normals.append(copy.deepcopy(normals))

        # get (a,b,c)s of the lines of the ego vehicle
        ego_lines = []
        cx,cy = desc.cur_vehicle_state.vehicle_location.x, desc.cur_vehicle_state.vehicle_location.y
        l,w = desc.cur_vehicle_state.length, desc.cur_vehicle_state.width
        ego_bb = [[cx+l/2,cy-w/2], [cx+l/2, cy+w/2], [cx-l/2, cy+w/2], [cx-l/2, cy-w/2]]
        for i in range(0,4):
            if i<2:
                x1,y1 = ego_bb[i%4]
                x2,y2 = ego_bb[(i+1)%4]
            else:
                x2,y2 = ego_bb[i%4]
                x1,y1 = ego_bb[(i+1)%4]
            a = (y2-y1)
            b = -(x2-x1)
            c = (y1*(x2-x1)-(y2-y1)*x1)
            norm = np.linalg.norm([a,b])
            a, b, c = a/norm, b/norm, c/norm
            ego_lines.append([a,b,c])
        # find closest distance
        closest = 1e5
        for bb,normals in zip(NPC_bb,NPC_normals):
            distance = self.project_and_calculate(bb,normals,ego_bb,ego_lines)
            if closest > distance:
                closest = distance
        return closest

    def position_cost(self):
        r_inv = 1/self.max_reward
        cost = max(0,1/(self.closest_dist+r_inv)-1/(self.closest_dist+r_inv))
        return self.k_pos*cost

    @abstractmethod
    def update(self,desc,action):
        return

    @abstractmethod
    def get_reward(self,desc,action):
        return
    
    @abstractmethod
    def reset(self):
        self.closest_dist=1e5

# Reward calculation functionality
class LaneChangeReward(Reward):
    '''
    ^x
    |
    |---->y
    '''
    def __init__(self):
        super().__init__()
        self.position_lb = 1
    # ---------------------------------HELPERS-----------------------------------------#
    def get_velocity_error(self,veh_speed):
        if veh_speed < self.min_vel:
            return self.p_vel*abs(veh_speed-self.min_vel)
        elif veh_speed > self.max_vel:
            return self.p_vel*abs(veh_speed-self.max_vel)
        else:
            return 0
    
    def vel_cost(self):
        return self.k_vel*self.cum_vel_err

    def position_cost(self):
        r_inv = 1/self.max_reward
        cost = max(0,1/(self.closest_dist+r_inv)-1/(self.position_lb+r_inv))
        return self.k_pos*cost
    
    def lane_change_reward(self,desc,action):
        if desc.reward.collision:
            return -self.max_reward
        else:
            reward=0
            reward+=self.max_reward
            reward-=self.vel_cost()
            reward-=self.position_cost()
            return reward

    def speed_reward(self,desc,action):
        reward=0
        reward-=self.vel_cost()
        reward-=self.position_cost()
        # reset at end of each action to keep negative reward bounded
        self.cum_vel_err=0
        return reward

    # ---------------------------------INTERFACES-----------------------------------------#
    def get_reward(self,desc,action):
        if action==RLDecision.SWITCH_LANE.value:
            # call lane change reward function
            return self.lane_change_reward(desc,action)
        elif action==RLDecision.CONSTANT_SPEED.value:
            # call constant speed reward functon
            return self.speed_reward(desc,action)
        elif action==RLDecision.ACCELERATE.value:
            # call accelerate reward function
            return self.speed_reward(desc,action)
        elif action==RLDecision.DECELERATE.value:
            # call decelerate reward function
            return self.speed_reward(desc,action)
    
    def update(self,desc,action):
        cur_dist = self.get_closest_distance(desc)
        if self.closest_dist > cur_dist:
            self.closest_dist=cur_dist
        self.cum_vel_err+=self.get_velocity_error(desc.cur_vehicle_state.vehicle_speed)

    def reset(self):
        self.closest_dist=1e5

class PedestrianReward(Reward):
    def __init__(self):
        super().__init__()
    # ---------------------------------HELPERS-----------------------------------------#
    def speed_reward(self,desc, action):
        reward=0
        self.position_lb = desc.nearest_pedestrian.radius
        ped_vehicle = VehicleState()
        relative_pose = ped_vehicle
        if desc.nearest_pedestrian.exist:
            ped_vehicle.vehicle_location = desc.nearest_pedestrian.pedestrian_location
            ped_vehicle.vehicle_speed = desc.nearest_pedestrian.pedestrian_speed
            relative_pose = convert_to_local(desc.cur_vehicle_state,ped_vehicle)
            # print("CUR")
            # print(desc.cur_vehicle_state)
            # print("PED")
            # print(ped_vehicle)
            # print("RELATIVE")
            # print(relative_pose)
        # check if pedestrian collided
        if desc.reward.collision:
            return-self.max_reward
        # check if pedestrian avoided
        elif desc.nearest_pedestrian.exist and relative_pose.vehicle_location.x < -1:
            reward=self.max_reward
        # add costs of overspeeding
        reward-=self.vel_cost()
        reward-=self.position_cost()
        self.reset()
        return reward

    def vel_cost(self):
        return self.k_vel*self.cum_vel_err
    
    def position_cost(self):
        r_inv = 1/self.max_reward
        cost = max(0,1/(self.closest_dist+r_inv)-1/(self.position_lb+r_inv))
        return self.k_pos*cost
        
    def get_velocity_error(self,veh_speed):
        if veh_speed > self.max_vel:
            return self.p_vel*abs(veh_speed-self.max_vel)
        else:
            return 0
    
    def get_closest_distance(self,desc):
        '''
        1          2 (+)         3
            +-------------+
            |        (-)  |
        4(-)|(+)   0   (-)| (+)  5
            |      (+)    |
            +-------------+
        6          7 (-)         8
        '''
        x = desc.nearest_pedestrian.pedestrian_location.x
        y = desc.nearest_pedestrian.pedestrian_location.x
        point=np.array([x,y,1])
        cx,cy = desc.cur_vehicle_state.vehicle_location.x, desc.cur_vehicle_state.vehicle_location.y
        radius = desc.nearest_pedestrian.radius
        l,w = desc.cur_vehicle_state.length, desc.cur_vehicle_state.width
        # bloat th corners
        l+=2*radius
        w+=2*radius
        ego_bb = [[cx+l/2,cy-w/2], [cx+l/2, cy+w/2], [cx-l/2, cy+w/2], [cx-l/2, cy-w/2]]
        ego_lines = []
        mapping = {
            (0,0,1,1):0,
            (1,0,1,0):1,
            (1,0,1,1):2,
            (1,1,1,1):3,
            (0,0,1,0):4,
            (0,1,1,1):5,
            (0,0,0,0):6,
            (0,0,0,1):7,
            (0,1,0,1):8
        }
        key = []
        for i in range(0,4):
            if i<2:
                x1,y1 = ego_bb[i%4]
                x2,y2 = ego_bb[(i+1)%4]
            else:
                x2,y2 = ego_bb[i%4]
                x1,y1 = ego_bb[(i+1)%4]
            a = (y2-y1)
            b = -(x2-x1)
            c = (y1*(x2-x1)-(y2-y1)*x1)
            norm = np.linalg.norm([a,b])
            a, b, c = a/norm, b/norm, c/norm
            ego_lines.append([a,b,c])
            dist = a*x+b*y+c
            k = 0 if dist < 0 else 1
            key.append(k)
        region = mapping[tuple(key)]
        closest = None
        if region == 0:
            closest = 0
        elif region == 1:
            closest = np.linalg.norm(point-np.array(ego_bb[0]+[1]))
        elif region == 2:
            closest = np.dot(point, ego_lines[0])
        elif region == 3:
            closest = np.linalg.norm(point-np.array(ego_bb[1]+[1]))
        elif region == 4:
            closest = abs(np.dot(point, ego_lines[3]))
        elif region == 5:
            closest = np.dot(point, ego_lines[1])
        elif region == 6:
            closest = np.linalg.norm(point-np.array(ego_bb[3]+[1]))
        elif region == 7:
            closest = abs(np.dot(point, ego_lines[2]))
        elif region == 8:
            closest = np.linalg.norm(point-np.array(ego_bb[2]+[1]))
        # print("Closest distance is ",closest)
        # print("Cur Vehicle", desc.cur_vehicle_state)
        # print("Pedestrian", desc.nearest_pedestrian)
        return closest

    # ---------------------------------INTERFACES-----------------------------------------#
    def get_reward(self,desc,action):
        print("Action",action)
        if action==RLDecision.CONSTANT_SPEED.value:
            # call constant speed reward functon
            return self.speed_reward(desc,action)
        elif action==RLDecision.ACCELERATE.value:
            # call accelerate reward function
            return self.speed_reward(desc,action)
        elif action==RLDecision.DECELERATE.value:
            # call decelerate reward function
            return self.speed_reward(desc,action)
        # reset closest distance at each action end
    
    def update(self,desc,action):
        if desc.nearest_pedestrian.exist:
            cur_dist = self.get_closest_distance(desc)
            if self.closest_dist > cur_dist:
                self.closest_dist=cur_dist
        self.cum_vel_err+=self.get_velocity_error(desc.cur_vehicle_state.vehicle_speed)

    def reset(self):
        self.closest_dist=1e5


def reward_selector(event):
    if event is Scenario.PEDESTRIAN:
        return PedestrianReward()
    elif event is Scenario.LANE_CHANGE:
        return LaneChangeReward()