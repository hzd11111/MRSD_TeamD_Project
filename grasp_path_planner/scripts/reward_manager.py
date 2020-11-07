import numpy as np
from abc import ABC, abstractmethod
# ROS Packages
from grasp_path_planner.msg import VehicleMsg
# other packages
from options import Scenario, RLDecision, TrafficLightStatus


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

    @abstractmethod
    def update(self, desc, action):
        """
        Meant to update costs and other intermediate variables
        """
        return

    @abstractmethod
    def get_reward(self, desc, action):
        """
        Gives the cost of taking action
        """
        return

    @abstractmethod
    def reset(self):
        """
        Resets environment
        """
        self.closest_dist = 1e5
    
    @abstractmethod
    def is_success(self, env_desc):
        return

    def calculate_line_coefficients(self, point1, point2):
        """
        Given two points gets the a,b,c values of the line equation
        """
        delta_x = point2[0] - point1[0]
        delta_y = point2[1] - point1[1]
        return [delta_y, -delta_x, point1[1] * delta_x - point1[0] * delta_y]

    def project_point_on_line(self, px, py, a, b, c):
        """
        Given a point and coefficients of a line equation
        gets the projection of the point on the line
        """
        k = b * px - a * py
        y = -(a * k + b * c) / (a**2 + b**2)
        x = (-c / a) - (b * y) / a if a > 0 else px
        return [x, y]


# Reward calculation functionality
class LaneChangeReward(Reward):
    '''
    ^x
    |
    |---->y
    '''

    def __init__(self):
        super().__init__()

    # ---------------------------------HELPERS-----------------------------------------#

    def calculate_bounding_box(self, cx: float, cy: float, theta: float, length: float, width: float):
        """
        Assumes angles are proper and not inverted.
        Gets the coordinates of the vehicle bounding box in global frame
        """
        bb_local = [[length / 2, -width / 2, 1],
                    [length / 2, width / 2, 1],
                    [-length / 2, width / 2, 1],
                    [-length / 2, -width / 2, 1]]
        # Create a transformation matrix, first translate and then rotate
        H_Rot = np.eye(3)
        H_Rot[-1, -1] = 1
        H_Rot[0, -1] = 0
        H_Rot[1, -1] = 0
        H_Rot[0, 0] = np.cos(theta)
        H_Rot[0, 1] = -np.sin(theta)
        H_Rot[1, 0] = np.sin(theta)
        H_Rot[1, 1] = np.cos(theta)
        H_trans = np.eye(3)
        H_trans[0, -1] = cx
        H_trans[1, -1] = cy
        H = np.matmul(H_trans, H_Rot)
        # Multiply transformation matrix and create a list of coords
        bb_coords = []
        for local_coord in bb_local:
            global_x, global_y = np.matmul(H, local_coord)[:2]
            bb_coords.append([global_x, global_y])
        return bb_coords

    def get_distance_between(self, bb1, bb2):
        """
        Gets the closest distane between two bounding boxes by applying SAT theorem
        """
        all_normals = []
        closest_dist = 0

        for i in range(4):
            all_normals.append(self.calculate_line_coefficients(bb1[i], bb1[(i + 1) % 4]))
        for i in range(4):
            all_normals.append(self.calculate_line_coefficients(bb2[i], bb2[(i + 1) % 4]))
        # project all points on these normals and get distance
        for normal in all_normals:
            bb1_projections = []
            bb2_projections = []
            # Let us project on normal centered at origin
            # ax+by+c = 0 is equivalent t0 bx+ay = 0
            for point in bb1:
                bb1_projections.append(self.project_point_on_line(point[0], point[1], normal[0], normal[1], normal[2]))
            for point in bb2:
                bb2_projections.append(self.project_point_on_line(point[0], point[1], normal[0], normal[1], normal[2]))

            # get the distance between these projections
            dist = 0
            if (abs(normal[0]) < 1e-3):
                # sort along x axis as y is constant
                bb1_projections = sorted(bb1_projections, key=lambda b: b[0])
                bb2_projections = sorted(bb2_projections, key=lambda b: b[0])
                # get the distance between sorted projections
                left_bb1, right_bb1 = bb1_projections[0], bb1_projections[-1]
                left_bb2, right_bb2 = bb2_projections[0], bb2_projections[-1]

                if (left_bb1[0] < left_bb2[0]) and (right_bb1[0] < left_bb2[0]):
                    dist = np.linalg.norm([right_bb1[0] - left_bb2[0], right_bb1[1] - left_bb2[1]])
                elif (left_bb2[0] < left_bb1[0]) and (right_bb2[0] < left_bb1[0]):
                    dist = np.linalg.norm([right_bb2[0] - left_bb1[0], right_bb2[1] - left_bb1[1]])
                else:
                    dist = 0

            else:
                # sort along y axis
                bb1_projections = sorted(bb1_projections, key=lambda b: b[1])
                bb2_projections = sorted(bb2_projections, key=lambda b: b[1])

                # get distance between sorted projections
                left_bb1, right_bb1 = bb1_projections[0], bb1_projections[-1]
                left_bb2, right_bb2 = bb2_projections[0], bb2_projections[-1]

                if (left_bb1[1] < left_bb2[1]) and (right_bb1[1] < left_bb2[1]):
                    dist = np.linalg.norm([right_bb1[0] - left_bb2[0], right_bb1[1] - left_bb2[1]])
                elif (left_bb2[1] < left_bb1[1]) and (right_bb2[1] < left_bb1[1]):
                    dist = np.linalg.norm([right_bb2[0] - left_bb1[0], right_bb2[1] - left_bb1[1]])
                else:
                    dist = 0

            # update the closest distance with max value
            if closest_dist < dist:
                closest_dist = dist

        return closest_dist

    def get_closest_distance(self, desc):
        """
        Get the closest distance between the ego vehicle and vehicles in the neighboring lane
        """
        NPCs = desc.adjacent_lane_vehicles
        NPC_bb = []
        # Get the bounding boxes for each vehicle
        for vehicle in NPCs:
            NPC_bb.append(self.calculate_bounding_box(vehicle.vehicle_location.x,
                                                      vehicle.vehicle_location.y,
                                                      vehicle.vehicle_location.theta,
                                                      vehicle.length,
                                                      vehicle.width))
        # Get the bounding boxes for ego vehicle
        ego_bb = self.calculate_bounding_box(desc.cur_vehicle_state.vehicle_location.x,
                                             desc.cur_vehicle_state.vehicle_location.y,
                                             desc.cur_vehicle_state.vehicle_location.theta,
                                             desc.cur_vehicle_state.length,
                                             desc.cur_vehicle_state.width)

        # Get the closest distance for each vehicle
        closest_vehicle_dist = 1e5
        for vehicle_bb in NPC_bb:
            dist = self.get_distance_between(ego_bb, vehicle_bb)
            if (closest_vehicle_dist > dist):
                closest_vehicle_dist = dist
        return closest_vehicle_dist

    def position_cost(self):
        """
        Gets the position cost based on how close ego vehicle is to other actors
        """
        r_inv = 1 / self.max_reward
        cost = max(0, 1 / (self.closest_dist + r_inv) - 1 / (self.min_dist + r_inv))
        return self.k_pos * cost

    def get_velocity_error(self, veh_speed):
        """
        Gets the error for over speeding or slowing down too much
        """
        if veh_speed < self.min_vel:
            return self.p_vel * abs(veh_speed - self.min_vel)
        elif veh_speed > self.max_vel:
            return self.p_vel * abs(veh_speed - self.max_vel)
        else:
            return 0

    def vel_cost(self):
        """
        Gets the cumulative velocity cost
        """
        return self.k_vel * self.cum_vel_err

    def lane_change_reward(self, desc, action):
        """
        Calculates the reward/penalty for the episode of lane change
        """
        if desc.reward.collision:
            return -self.max_reward
        else:
            reward = 0
            reward += self.max_reward
            reward -= self.vel_cost()
            reward -= self.position_cost()
            return reward

    def speed_reward(self, desc, action):
        """
        Gets the cost of actions like acclerate decelerate constant speed
        """
        reward = 0
        reward -= self.vel_cost()
        reward -= self.position_cost()
        # reset at end of each action to keep negative reward bounded
        self.cum_vel_err = 0
        return reward

    # ---------------------------------INTERFACES-----------------------------------------#
    def get_reward(self, desc, action):
        """
        Implementing interface
        """
        if action == RLDecision.SWITCH_LANE:
            # call lane change reward function
            return self.lane_change_reward(desc, action)
        elif action == RLDecision.CONSTANT_SPEED:
            # call constant speed reward functon
            return self.speed_reward(desc, action)
        elif action == RLDecision.ACCELERATE:
            # call accelerate reward function
            return self.speed_reward(desc, action)
        elif action == RLDecision.DECELERATE:
            # call decelerate reward function
            return self.speed_reward(desc, action)

    def update(self, desc, action):
        """
        Implementing Interface
        """
        cur_dist = self.get_closest_distance(desc)
        # print("Closest Distance Now:", cur_dist)
        if self.closest_dist > cur_dist:
            self.closest_dist = cur_dist
        self.cum_vel_err += self.get_velocity_error(desc.cur_vehicle_state.vehicle_speed)

    def reset(self):
        """
        Implementing Interface
        """
        self.closest_dist = 1e5
        self.cum_vel_err = 0


class PedestrianReward(Reward):
    def __init__(self):
        super().__init__()
        self.max_reward = 1

    # ---------------------------------HELPERS-----------------------------------------#
    def speed_reward(self, desc, action):
        """
        Gets the reward/penalty for accelerate, decelerate, constant speed
        """
        reward = 0
        self.min_dist = desc.nearest_pedestrian.radius
        ped_vehicle = VehicleState()
        relative_pose = ped_vehicle
        if desc.nearest_pedestrian.exist:
            ped_vehicle.vehicle_location = desc.nearest_pedestrian.pedestrian_location
            ped_vehicle.vehicle_speed = desc.nearest_pedestrian.pedestrian_speed
            relative_pose = convertToLocal(desc.cur_vehicle_state, ped_vehicle)

        # check if pedestrian collided
        if desc.reward.collision:
            return -self.max_reward
        # check if pedestrian avoided
        elif desc.nearest_pedestrian.exist and relative_pose.vehicle_location.x < -10:
            reward = self.max_reward
        # add costs of overspeeding
        reward -= self.vel_cost()
        reward -= self.position_cost()
        self.reset()
        return reward

    def vel_cost(self):
        """
        Gets the cumulative velocity cost for making velocity errors
        """
        return self.k_vel * self.cum_vel_err

    def position_cost(self):
        """
        Gets the cost for going too close to other actors
        """
        r_inv = 1 / self.max_reward
        cost = max(0, 1 / (self.closest_dist + r_inv) - 1 / (self.min_dist + r_inv))
        return self.k_pos * cost

    def get_velocity_error(self, veh_speed):
        """
        Computes the velocity error
        """
        if veh_speed > self.max_vel:
            return self.p_vel * abs(veh_speed - self.max_vel)
        else:
            return 0

    def get_closest_distance(self, desc):
        """
        Gets the distance between the ego vehicle and pedestrian
        """
        # get the transform for converting global coordinate to local coordinate
        theta = desc.cur_vehicle_state.vehicle_location.theta
        cx = desc.cur_vehicle_state.vehicle_location.x
        cy = desc.cur_vehicle_state.vehicle_location.y
        H_Rot = np.eye(3)
        H_Rot[-1, -1] = 1
        H_Rot[0, -1] = 0
        H_Rot[1, -1] = 0
        H_Rot[0, 0] = np.cos(theta)
        H_Rot[0, 1] = np.sin(theta)
        H_Rot[1, 0] = -np.sin(theta)
        H_Rot[1, 1] = np.cos(theta)
        H_trans = np.eye(3)
        H_trans[0, -1] = -cx
        H_trans[1, -1] = -cy
        H = np.matmul(H_Rot, H_trans)

        # inflate legth and width with radius
        r = desc.nearest_pedestrian.radius
        l = desc.cur_vehicle_state.length + 2 * r
        w = desc.cur_vehicle_state.width + 2 * r

        # get the pedestrian in local coordinate frame
        ped_local = np.matmul(H,
                              [desc.nearest_pedestrian.pedestrian_location.x,
                               desc.nearest_pedestrian.pedestrian_location.y,
                               1])

        # identify which point it is closest to
        bb_local = [[l / 2, -w / 2],
                    [l / 2, w / 2],
                    [-l / 2, w / 2],
                    [-l / 2, -w / 2]]
        dist = 1e5
        closest_idx = -1
        for i, point in enumerate(bb_local):
            cur_dist = np.linalg.norm([point[0] - ped_local[0], point[1] - ped_local[1]])
            if cur_dist < dist:
                dist = cur_dist
                closest_idx = i

        # check if distance to line with closest_idx is closer
        a, b, c = self.calculate_line_coefficients(bb_local[closest_idx], bb_local[(closest_idx + 1) % 4])
        projected_ped = self.project_point_on_line(ped_local[0], ped_local[1], a, b, c)
        projected_dist = np.linalg.norm([projected_ped[0] - ped_local[0], ped_local[1] - projected_ped[1]])
        if (projected_dist < dist):
            dist = projected_dist

        a, b, c = self.calculate_line_coefficients(bb_local[closest_idx], bb_local[(closest_idx - 1) % 4])
        projected_ped = self.project_point_on_line(ped_local[0], ped_local[1], a, b, c)
        projected_dist = np.linalg.norm([projected_ped[0] - ped_local[0], ped_local[1] - projected_ped[1]])
        if (projected_dist < dist):
            dist = projected_dist

        # return the dist
        return dist
    # ---------------------------------INTERFACES-----------------------------------------#

    def get_reward(self, desc, action):
        """
        Implementing Interface
        """
        if action == RLDecision.CONSTANT_SPEED:
            # call constant speed reward functon
            return self.speed_reward(desc, action)
        elif action == RLDecision.ACCELERATE:
            # call accelerate reward function
            return self.speed_reward(desc, action)
        elif action == RLDecision.DECELERATE:
            # call decelerate reward function
            return self.speed_reward(desc, action)
        # reset closest distance at each action end

    def update(self, desc, action):
        """
        Implementing Interface
        """
        if desc.nearest_pedestrian.exist:
            cur_dist = self.get_closest_distance(desc)
            if self.closest_dist > cur_dist:
                self.closest_dist = cur_dist
        self.cum_vel_err += self.get_velocity_error(desc.cur_vehicle_state.vehicle_speed)

    def reset(self):
        """
        Implementing Interface
        """
        self.closest_dist = 1e5
        self.cum_vel_err = 0


# Plain Reward Class
class PlainRewardWithLight(Reward):
    '''
    Implements positional costs and basic required interfaces for all rewards
    '''

    def __init__(self):
        super().__init__()
        self.max_reward = 1

    def update(self, desc, action):
        """
        Meant to update costs and other intermediate variables
        """
        return

    def ran_red_light(self, env_desc):
        flag = env_desc.cur_vehicle_state.traffic_light_status is TrafficLightStatus.RED and \
            env_desc.cur_vehicle_state.traffic_light_stop_distance >= 0 and \
            env_desc.cur_vehicle_state.traffic_light_stop_distance <= 2
        if flag:
            print("Red light ran", flag)
        return flag

    def get_reward(self, desc, action):
        """
        Gives the cost of taking action
        """
        reward = 0
        # print("Action progress is ", desc.reward_info.action_progress)
        if desc.reward_info.collision or \
                (desc.reward_info.time_elapsed > 80) or \
                self.ran_red_light(desc):
            reward = reward - 1
        elif desc.reward_info.path_planner_terminate:
            reward += desc.reward_info.action_progress
        print("Reward is ", reward)
        return reward

    def reset(self):
        """
        Resets environment
        """
        return 
    
    def is_success(self, desc):
        return not (desc.reward_info.collision or \
                (desc.reward_info.time_elapsed > 80) or \
                self.ran_red_light(desc))


# Plain Lane switch reward
class PlainLaneChange(Reward):
    '''
    Implements positional costs and basic required interfaces for all rewards
    '''

    def __init__(self):
        super().__init__()
        self.max_reward = 1

    def update(self, desc, action):
        """
        Meant to update costs and other intermediate variables
        """
        return

    def get_reward(self, desc, action):
        """
        Gives the cost of taking action
        """
        reward = 0
        if desc.reward_info.collision or \
                (desc.reward_info.time_elapsed > 80) or \
                desc.reward_info.lane_switch_failure_terminate:
            reward = reward - 1
        elif desc.reward_info.path_planner_terminate:
            reward += desc.reward_info.action_progress
        print("Reward is ", reward)
        return reward

    def reset(self):
        """
        Resets environment
        """
        return
    
    def is_success(self, desc):
        return not(desc.reward_info.collision or \
                (desc.reward_info.time_elapsed > 80) or \
                desc.reward_info.lane_switch_failure_terminate)


class PlainReward(Reward):
    '''
    Implements positional costs and basic required interfaces for all rewards
    '''

    def __init__(self):
        super().__init__()
        self.max_reward = 1

    def update(self, desc, action):
        """
        Meant to update costs and other intermediate variables
        """
        return

    def get_reward(self, desc, action):
        """
        Gives the cost of taking action
        """
        reward = 0
        # print("Action progress is ", desc.reward_info.action_progress)
        if desc.reward_info.collision or \
                (desc.reward_info.time_elapsed > 80):
            reward = reward - 1
        elif desc.reward_info.path_planner_terminate:
            reward += desc.reward_info.action_progress
        print("Reward is ", reward)
        return reward

    def reset(self):
        """
        Resets environment
        """
        return
    
    def is_success(self, desc):
        return not (desc.reward_info.collision or \
                (desc.reward_info.time_elapsed > 80))

def reward_selector(event):
    """
    Select reward manager based on event
    """
    if event is Scenario.SWITCH_LANE_RIGHT or event is Scenario.SWITCH_LANE_LEFT:
        return PlainLaneChange()
    elif event is Scenario.RIGHT_TURN or event is Scenario.LANE_FOLLOWING:
        return PlainReward()
    else:
        return PlainRewardWithLight()
