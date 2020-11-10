#!/usr/bin/env python
import sys
sys.path.append("../../carla_utils/utils")
from utility import *
from functional_utility import *
from options import RLDecision
import copy

class TrajGenerator:
    SAME_POSE_THRESHOLD = 2
    SAME_POSE_LOWER_THRESHOLD = 0.02
    MAX_GLOBAL_SPEED = 20
    MAX_STRAIGHT_SPEED = 35

    # constructor
    def __init__(self, traj_parameters):
        self.traj_parameters = traj_parameters
        self.current_action = RLDecision.NO_ACTION
        self.generated_path = []
        self.path_pointer = 0
        self.action_start_time = 0
        self.start_speed = 0
        self.global_path_pointer = 0
        self.reset()

    # reset the traj generator with
    def reset(self, cur_act=RLDecision.NO_ACTION, start_speed=0, action_start_time=0, complete_reset = False):
        self.current_action = cur_act
        self.generated_path = []
        self.path_pointer = 0
        self.action_start_time = action_start_time
        self.start_speed = start_speed
        if complete_reset:
            self.global_path_pointer = 0


    def newActionType(self, rl_decision):
        if self.current_action == RLDecision.NO_ACTION:
            return rl_decision
        return self.current_action

    def trajPlan(self, rl_decision, env_desc, chosen_scenario = None, dummy_action = False):


        # check the validity of env_desc
        # import ipdb; ipdb.set_trace()
        if rl_decision is None or dummy_action:
            path_plan = self.tempConstSpeed(env_desc)
            if chosen_scenario is not None:
                path_plan.scenario_chosen = chosen_scenario
            return path_plan.toRosMsg()
        # continue the previous action if it hasn't ended
        action_to_perform = self.newActionType(rl_decision)

        # plan trajectory switch cases
        if action_to_perform == RLDecision.CONSTANT_SPEED:
            path_plan = self.constSpeedTraj(env_desc)
            if chosen_scenario is not None:
                path_plan.scenario_chosen = chosen_scenario
            return path_plan.toRosMsg()
        elif action_to_perform == RLDecision.ACCELERATE:
            path_plan = self.accelerateTraj(env_desc)
            if chosen_scenario is not None:
                path_plan.scenario_chosen = chosen_scenario
            return path_plan.toRosMsg()
        elif action_to_perform == RLDecision.DECELERATE:
            path_plan = self.decelerateTraj(env_desc)
            if chosen_scenario is not None:
                path_plan.scenario_chosen = chosen_scenario
            return path_plan.toRosMsg()
        elif (action_to_perform == RLDecision.SWITCH_LANE_LEFT) or \
                (action_to_perform == RLDecision.SWITCH_LANE_RIGHT):
            path_plan = self.laneChangeTraj(env_desc, rl_decision)
            if chosen_scenario is not None:
                path_plan.scenario_chosen = chosen_scenario
            return path_plan.toRosMsg()
        elif action_to_perform == RLDecision.GLOBAL_PATH_CONSTANT_SPEED:
            path_plan = self.globalConstSpeedTraj(env_desc)
            if chosen_scenario is not None:
                path_plan.scenario_chosen = chosen_scenario
            return path_plan.toRosMsg()
        elif action_to_perform == RLDecision.GLOBAL_PATH_ACCELERATE:
            path_plan = self.globalAccelerateTraj(env_desc)
            if chosen_scenario is not None:
                path_plan.scenario_chosen = chosen_scenario
            return path_plan.toRosMsg()
        elif action_to_perform == RLDecision.GLOBAL_PATH_DECELERATE:
            path_plan = self.globalDecelerateTraj(env_desc)
            if chosen_scenario is not None:
                path_plan.scenario_chosen = chosen_scenario
            return path_plan.toRosMsg()
        elif action_to_perform == RLDecision.STOP:
            path_plan = self.stop(env_desc)
            if chosen_scenario is not None:
                path_plan.scenario_chosen = chosen_scenario
            return path_plan.toRosMsg()
        else:
            print("RLDecision ERROR:", action_to_perform)

    def stop(self, sim_data):
        if not self.current_action == RLDecision.STOP:
            self.reset(RLDecision.STOP, sim_data.cur_vehicle_state.speed,
                       sim_data.reward_info.time_elapsed)
        # current vehicle state
        curr_vehicle = sim_data.cur_vehicle_state
        curr_vehicle_global_pose = curr_vehicle.location_global

        # determine the closest next pose in the global path
        global_path = sim_data.global_path
        global_path_points = global_path.path_points
        tracking_pose = False
        tracking_pose_ind = False
        for point_ind, path_point in enumerate(global_path_points):
            single_pose = path_point.global_pose
            if single_pose.isInfrontOf(curr_vehicle_global_pose) and \
                   single_pose.distance(curr_vehicle_global_pose) > TrajGenerator.SAME_POSE_LOWER_THRESHOLD and \
                single_pose.distance(curr_vehicle_global_pose) < 1.5:
                tracking_pose = single_pose
                tracking_pose_ind = point_ind
                break

        # determine the speed needed
        action_end = not (tracking_pose and not tracking_pose_ind >= len(global_path_points) - 1)
        if not action_end:
            # determine the distance till the end of path
            distance = curr_vehicle_global_pose.distance(global_path_points[-1].global_pose)
            target_speed = 20 - 20 / distance
            target_pose = tracking_pose
        else:
            target_speed = 0
            next_point_frenet = Frenet()
            next_point_frenet.x = 1.
            target_pose = sim_data.current_lane.frenetToGlobal(next_point_frenet)

        new_path_plan = PathPlan()
        new_path_plan.tracking_pose = target_pose
        new_path_plan.reset_sim = False
        new_path_plan.tracking_speed = target_speed
        new_path_plan.end_of_action = action_end
        new_path_plan.action_progress = 0.5
        new_path_plan.path_planner_terminate = action_end

        # add future poses ToDo
        new_path_plan.future_poses = []

        if new_path_plan.end_of_action:
            self.reset()
        return new_path_plan



    # find the next lane point of the current vehicle
    def findNextLanePose(self, curr_vehicle, lane_point_array):
        vehicle_frenet = curr_vehicle.location_frenet
        vehicle_frenet_x = vehicle_frenet.x

        for lane_waypoint in lane_point_array:
            lane_point_frenet = lane_waypoint.frenet_pose
            lane_point_frenet_x = lane_point_frenet.x
            frenet_x_diff = lane_point_frenet_x - vehicle_frenet_x
            if frenet_x_diff >= TrajGenerator.SAME_POSE_LOWER_THRESHOLD:
                return lane_waypoint
        return lane_point_array[-1]

    def globalConstSpeedTraj(self, sim_data):
        # check if this is a new action
        if not self.current_action == RLDecision.GLOBAL_PATH_CONSTANT_SPEED:
            self.reset(RLDecision.GLOBAL_PATH_CONSTANT_SPEED,
                       sim_data.cur_vehicle_state.speed,
                       sim_data.reward_info.time_elapsed)

        # current simulation time
        new_sim_time = sim_data.reward_info.time_elapsed

        # current vehicle state
        curr_vehicle = sim_data.cur_vehicle_state
        curr_vehicle_global_pose = curr_vehicle.location_global

        # determine the closest next pose in the global path
        global_path = sim_data.intersection_global_path
        global_path_points = global_path.path_points
        tracking_pose = False
        tracking_pose_ind = False
        while self.global_path_pointer < len(global_path_points):
            single_pose = global_path_points[self.global_path_pointer].global_pose
            tracking_pose = single_pose
            if single_pose.isInfrontOf(curr_vehicle_global_pose) and \
                single_pose.distance(curr_vehicle_global_pose) > TrajGenerator.SAME_POSE_LOWER_THRESHOLD:
                break
            self.global_path_pointer += 1

        # determine the action progress
        action_progress = (new_sim_time - self.action_start_time) / self.traj_parameters['action_duration']

        # determine if this is the end of an action
        end_of_action = False
        if action_progress >= 1.:
            end_of_action = True
            action_progress = 1.
            
        path_planner_terminate = False
        if self.global_path_pointer >= len(global_path_points):
            tracking_pose = global_path_points[-1].global_pose
            action_progress = 1.
            end_of_action = True
            path_planner_terminate = True
            print("Lane Navigation Terminated")

        new_path_plan = PathPlan()
        new_path_plan.tracking_pose = tracking_pose
        new_path_plan.reset_sim = False
        new_path_plan.tracking_speed = min(max(self.traj_parameters['min_speed'],self.start_speed), self.MAX_GLOBAL_SPEED)
        new_path_plan.end_of_action = end_of_action
        new_path_plan.action_progress = action_progress
        new_path_plan.path_planner_terminate = path_planner_terminate

        # add future poses ToDo
        new_path_plan.future_poses = []
        for i in range(tracking_pose_ind, self.global_path_pointer + 5):
            if i < len(global_path_points):
                new_path_plan.future_poses.append(global_path_points[i].global_pose)

        if end_of_action:
            self.reset()

        return new_path_plan

    def globalAccelerateTraj(self, sim_data):
        # check if this is a new action
        if not self.current_action == RLDecision.GLOBAL_PATH_ACCELERATE:
            self.reset(RLDecision.GLOBAL_PATH_ACCELERATE,
                       sim_data.cur_vehicle_state.speed,
                       sim_data.reward_info.time_elapsed)

        # current simulation time
        new_sim_time = sim_data.reward_info.time_elapsed

        # current vehicle state
        curr_vehicle = sim_data.cur_vehicle_state
        curr_vehicle_global_pose = curr_vehicle.location_global

        # determine the closest next pose in the global path
        global_path = sim_data.intersection_global_path
        global_path_points = global_path.path_points
        tracking_pose = False
        tracking_pose_ind = False
        while self.global_path_pointer < len(global_path_points):
            single_pose = global_path_points[self.global_path_pointer].global_pose
            tracking_pose = single_pose
            if single_pose.isInfrontOf(curr_vehicle_global_pose) and \
                single_pose.distance(curr_vehicle_global_pose) > TrajGenerator.SAME_POSE_LOWER_THRESHOLD:
                break
            self.global_path_pointer += 1

        # determine the action progress
        action_progress = (new_sim_time - self.action_start_time) / self.traj_parameters['action_duration']

        # determine if this is the end of an action
        end_of_action = False
        if action_progress >= 1.:
            end_of_action = True
            action_progress = 1.
            
        path_planner_terminate = False
        if self.global_path_pointer >= len(global_path_points):
            tracking_pose = global_path_points[-1].global_pose
            action_progress = 1.
            end_of_action = True
            path_planner_terminate = True
            print("Lane Navigation Terminated")

        new_path_plan = PathPlan()
        new_path_plan.tracking_pose = tracking_pose
        new_path_plan.reset_sim = False
        new_path_plan.tracking_speed = min(max(self.traj_parameters['min_speed'],self.start_speed + action_progress * self.traj_parameters['accelerate_amt']), self.MAX_GLOBAL_SPEED)
        new_path_plan.end_of_action = end_of_action
        new_path_plan.action_progress = action_progress
        new_path_plan.path_planner_terminate = path_planner_terminate

        # add future poses ToDo
        new_path_plan.future_poses = []
        for i in range(tracking_pose_ind, self.global_path_pointer + 5):
            if i < len(global_path_points):
                new_path_plan.future_poses.append(global_path_points[i].global_pose)

        if end_of_action:
            self.reset()

        return new_path_plan

    def globalDecelerateTraj(self, sim_data):
        # check if this is a new action
        if not self.current_action == RLDecision.GLOBAL_PATH_DECELERATE:
            self.reset(RLDecision.GLOBAL_PATH_DECELERATE,
                       sim_data.cur_vehicle_state.speed,
                       sim_data.reward_info.time_elapsed)

        # current simulation time
        new_sim_time = sim_data.reward_info.time_elapsed

        # current vehicle state
        curr_vehicle = sim_data.cur_vehicle_state
        curr_vehicle_global_pose = curr_vehicle.location_global

        # determine the closest next pose in the global path
        global_path = sim_data.intersection_global_path
        global_path_points = global_path.path_points
        tracking_pose = False
        tracking_pose_ind = False
        while self.global_path_pointer < len(global_path_points):
            single_pose = global_path_points[self.global_path_pointer].global_pose
            tracking_pose = single_pose
            if single_pose.isInfrontOf(curr_vehicle_global_pose) and \
                single_pose.distance(curr_vehicle_global_pose) > TrajGenerator.SAME_POSE_LOWER_THRESHOLD:
                break
            self.global_path_pointer += 1

        # if not tracking_pose:
        #     print("--------------- Global Path Error --------------")

        # determine the action progress
        action_progress = (new_sim_time - self.action_start_time) / self.traj_parameters['action_duration']

        # determine if this is the end of an action
        end_of_action = False
        if action_progress >= 1.:
            end_of_action = True
            action_progress = 1.
            
        path_planner_terminate = False
        if self.global_path_pointer >= len(global_path_points):
            tracking_pose = global_path_points[-1].global_pose
            action_progress = 1.
            end_of_action = True
            path_planner_terminate = True
            print("Lane Navigation Terminated")


        new_path_plan = PathPlan()
        new_path_plan.tracking_pose = tracking_pose
        new_path_plan.reset_sim = False
        new_path_plan.tracking_speed = min(max(self.traj_parameters['min_speed'],
                                           self.start_speed - action_progress * self.traj_parameters['decelerate_amt']), self.MAX_GLOBAL_SPEED)
        new_path_plan.end_of_action = end_of_action
        new_path_plan.action_progress = action_progress
        new_path_plan.path_planner_terminate = path_planner_terminate

        # add future poses ToDo
        new_path_plan.future_poses = []
        for i in range(tracking_pose_ind, self.global_path_pointer + 5):
            if i < len(global_path_points):
                new_path_plan.future_poses.append(global_path_points[i].global_pose)

        if end_of_action:
            self.reset()

        return new_path_plan

    def tempConstSpeed(self, sim_data):
        # determine the closest next pose in the current lane
        current_lane = sim_data.current_lane

        next_point_frenet = Frenet()
        next_point_frenet.x = 1.
        new_path_plan = PathPlan()
        new_path_plan.tracking_pose = current_lane.frenetToGlobal(next_point_frenet)
        new_path_plan.reset_sim = False
        new_path_plan.tracking_speed = max(self.traj_parameters['min_speed'], self.start_speed) + 1
        new_path_plan.end_of_action = True
        new_path_plan.action_progress = 1.
        new_path_plan.path_planner_terminate = False

        # add future poses
        new_path_plan.future_poses = []

        self.reset()
        return new_path_plan

    def constSpeedTraj(self, sim_data):
        # check if this is a new action
        if not self.current_action == RLDecision.CONSTANT_SPEED:
            self.reset(RLDecision.CONSTANT_SPEED,
                       sim_data.cur_vehicle_state.speed,
                       sim_data.reward_info.time_elapsed)

        # current simulation time
        new_sim_time = sim_data.reward_info.time_elapsed

        # current vehicle state
        curr_vehicle = sim_data.cur_vehicle_state

        # determine the closest next pose in the current lane
        current_lane = sim_data.current_lane

        # determine the action progress
        action_progress = (new_sim_time - self.action_start_time) / self.traj_parameters['action_duration']

        # determine if this is the end of an action
        end_of_action = False
        if action_progress >= 1.:
            end_of_action = True
            action_progress = 1.

        next_point_frenet = Frenet()
        next_point_frenet.x = 1.
        new_path_plan = PathPlan()
        new_path_plan.tracking_pose = current_lane.frenetToGlobal(next_point_frenet)
        new_path_plan.reset_sim = False
        new_path_plan.tracking_speed = max(self.traj_parameters['min_speed'],self.start_speed)
        new_path_plan.end_of_action = end_of_action
        new_path_plan.action_progress = action_progress
        new_path_plan.path_planner_terminate = False

        # add future poses
        new_path_plan.future_poses = []
        tracking_pose_frenet = next_point_frenet
        for ts in np.arange(self.traj_parameters['action_time_disc'], \
                self.traj_parameters['action_duration'] - (new_sim_time - self.action_start_time), \
                            self.traj_parameters['action_time_disc']):
            delta_x = ts * self.start_speed / 3.6
            new_frenet = copy.copy(tracking_pose_frenet)
            new_frenet.x += delta_x

            new_global_pose = current_lane.frenetToGlobal(new_frenet)
            new_path_plan.future_poses.append(new_global_pose)

        if end_of_action:
            self.reset()

        return new_path_plan

    def accelerateTraj(self, sim_data):

        # check if this is a new action
        if not self.current_action == RLDecision.ACCELERATE:
            self.reset(RLDecision.ACCELERATE,
                       sim_data.cur_vehicle_state.speed,
                       sim_data.reward_info.time_elapsed)

        # current simulation time
        new_sim_time = sim_data.reward_info.time_elapsed

        # current vehicle state
        curr_vehicle = sim_data.cur_vehicle_state

        # determine the closest next pose in the current lane
        current_lane = sim_data.current_lane

        # determine the action progress
        action_progress = (new_sim_time - self.action_start_time) / self.traj_parameters['action_duration']

        # determine if this is the end of an action
        end_of_action = False
        if action_progress >= 1.:
            end_of_action = True
            action_progress = 1.

        next_point_frenet = Frenet()
        next_point_frenet.x = 1.
        new_path_plan = PathPlan()
        new_path_plan.tracking_pose = current_lane.frenetToGlobal(next_point_frenet)
        new_path_plan.reset_sim = False
        new_path_plan.tracking_speed = min(max(self.traj_parameters['min_speed'],self.start_speed + action_progress * self.traj_parameters['accelerate_amt']), self.MAX_STRAIGHT_SPEED)
        new_path_plan.end_of_action = end_of_action
        new_path_plan.action_progress = action_progress
        new_path_plan.path_planner_terminate = False

        # add future poses
        new_path_plan.future_poses = []
        tracking_pose_frenet = next_point_frenet
        acc_per_sec = self.traj_parameters['accelerate_amt'] / self.traj_parameters['action_duration']
        for ts in np.arange(self.traj_parameters['action_time_disc'], \
                self.traj_parameters['action_duration'] - (new_sim_time - self.action_start_time), \
                            self.traj_parameters['action_time_disc']):
            delta_x = (ts * (self.start_speed + action_progress * self.traj_parameters['accelerate_amt']) + \
                            acc_per_sec * (ts ** 2.) / 2.) / 3.6
            new_frenet = copy.copy(tracking_pose_frenet)
            new_frenet.x += delta_x
            new_global_pose = current_lane.frenetToGlobal(new_frenet)
            new_path_plan.future_poses.append(new_global_pose)

        if end_of_action:
            self.reset()

        return new_path_plan

    def decelerateTraj(self, sim_data):

        # check if this is a new action
        if not self.current_action == RLDecision.DECELERATE:
            self.reset(RLDecision.DECELERATE,
                       sim_data.cur_vehicle_state.speed,
                       sim_data.reward_info.time_elapsed)

        # current simulation time
        new_sim_time = sim_data.reward_info.time_elapsed

        # current vehicle state
        curr_vehicle = sim_data.cur_vehicle_state

        # determine the closest next pose in the current lane
        current_lane = sim_data.current_lane
        lane_point_array = current_lane.lane_points
        next_lane_point = self.findNextLanePose(curr_vehicle, lane_point_array)

        # determine the action progress
        action_progress = (new_sim_time - self.action_start_time) / self.traj_parameters['action_duration']

        # determine if this is the end of an action
        end_of_action = False
        if action_progress >= 1.:
            end_of_action = True
            action_progress = 1.

        next_point_frenet = Frenet()
        next_point_frenet.x = 1.
        new_path_plan = PathPlan()
        new_path_plan.tracking_pose = current_lane.frenetToGlobal(next_point_frenet)
        new_path_plan.reset_sim = False
        new_path_plan.tracking_speed = max(self.traj_parameters['min_speed'],
                                           self.start_speed - action_progress * self.traj_parameters['decelerate_amt'])
        new_path_plan.end_of_action = end_of_action
        new_path_plan.action_progress = action_progress
        new_path_plan.path_planner_terminate = False
        # add future poses
        new_path_plan.future_poses = []
        tracking_pose_frenet = next_point_frenet
        dec_per_sec = self.traj_parameters['decelerate_amt'] / self.traj_parameters['action_duration']
        for ts in np.arange(self.traj_parameters['action_time_disc'], \
                self.traj_parameters['action_duration'] - (new_sim_time - self.action_start_time), \
                            self.traj_parameters['action_time_disc']):
            delta_x = (ts * (self.start_speed - action_progress * self.traj_parameters['accelerate_amt']) -\
                            dec_per_sec * (ts**2.) / 2.) / 3.6
            new_frenet = copy.copy(tracking_pose_frenet)
            new_frenet.x += delta_x
            new_global_pose = current_lane.frenetToGlobal(new_frenet)
            new_path_plan.future_poses.append(new_global_pose)

        if end_of_action:
            self.reset()

        return new_path_plan

    def cubicSplineGen(self, lane_dist, v_cur):
        v_cur = v_cur / 3.6
        if v_cur < 5:
            v_cur = 5
        # determine external parameters
        w = lane_dist
        l = self.traj_parameters['lane_change_length']
        r = self.traj_parameters['lane_change_time_constant']
        tf = math.sqrt(l ** 2 + w ** 2) * r / v_cur

        # parameters for x
        dx = 0
        cx = v_cur
        ax = (2. * v_cur * tf - 2. * l) / (tf ** 3.)
        bx = -3. / 2 * ax * tf

        # parameters for y
        dy = 0
        cy = 0
        ay = -2. * w / (tf ** 3.)
        by = 3 * w / (tf ** 2.)

        # return result
        neutral_traj = []

        # time loop
        time_disc = self.traj_parameters['lane_change_time_disc']
        total_loop_count = int(tf / time_disc + 1)
        for i in range(total_loop_count):
            t = i * time_disc
            x_value = ax * (t ** 3.) + bx * (t ** 2.) + cx * t + dx
            y_value = -(ay * (t ** 3.) + by * (t ** 2.) + cy * t + dy)
            x_deriv = 3. * ax * (t ** 2.) + 2. * bx * t + cx
            y_deriv = -(3. * ay * (t ** 2.) + 2. * by * t + cy)
            theta = math.atan2(y_deriv, x_deriv)
            speed = math.sqrt(y_deriv ** 2. + x_deriv ** 2.)
            frenet_pose = Frenet()
            tracking_speed = speed * 3.6
            frenet_pose.x = x_value
            frenet_pose.y = y_value
            frenet_pose.theta = theta
            neutral_traj.append((frenet_pose, tracking_speed))

        return neutral_traj

    def laneChangeTraj(self, sim_data, rl_decision):
        # check if this is a new action
        if not self.current_action == rl_decision:
            self.reset(rl_decision,
                       sim_data.cur_vehicle_state.speed,
                       sim_data.reward_info.time_elapsed)

            # current simulation time
            new_sim_time = sim_data.reward_info.time_elapsed

            # current vehicle state
            curr_vehicle = sim_data.cur_vehicle_state

            # find the target lane
            target_lane = False
            if rl_decision == RLDecision.SWITCH_LANE_LEFT:
                for parallel_lane in sim_data.adjacent_lanes:
                    if parallel_lane.adjacent_lane and \
                        parallel_lane.left_to_the_current and \
                        parallel_lane.same_direction:
                        target_lane = parallel_lane
                        break
            else:
                for parallel_lane in sim_data.adjacent_lanes:
                    if parallel_lane.adjacent_lane and \
                        not parallel_lane.left_to_the_current and \
                        parallel_lane.same_direction:
                        target_lane = parallel_lane
                        break

            if target_lane == False:
                print("-----No lane found-----")
                raise

            # determine the closest next pose in the current lane
            current_lane = sim_data.current_lane
            #lane_point_array = current_lane.lane_points
            #next_lane_point = self.findNextLanePose(curr_vehicle, lane_point_array)

            # determine the lane distance
            lane_distance = target_lane.lane_distance

            # generate lane changing spline
            neutral_spline_left = self.cubicSplineGen(lane_distance, sim_data.cur_vehicle_state.speed)

            # append neutral spline to the nearest lane point and flip for a right turn
            x_offset = 0.5
            for spline_frenet in neutral_spline_left:
                spline_frenet[0].x += x_offset
                if rl_decision == RLDecision.SWITCH_LANE_RIGHT:
                    spline_frenet[0].y = -spline_frenet[0].y
                    spline_frenet[0].theta = -spline_frenet[0].theta

            #  make all lane points global pose
            global_spline = []
            for spline_frenet in neutral_spline_left:
                global_pose = current_lane.frenetToGlobal(spline_frenet[0])
                speed = spline_frenet[1]
                global_pose_speed = PoseSpeed(global_pose)
                global_pose_speed.speed = speed
                global_spline.append(global_pose_speed)

            self.generated_path = global_spline
            self.path_pointer = 0

        # current vehicle state
        curr_vehicle = sim_data.cur_vehicle_state
        curr_vehicle_global_pose = curr_vehicle.location_global

        # find the next tracking point
        while (self.path_pointer < len(self.generated_path)):
            # traj pose
            pose_speed = self.generated_path[self.path_pointer]

            if pose_speed.isInfrontOf(curr_vehicle_global_pose) and \
                    pose_speed.distance(curr_vehicle_global_pose) > TrajGenerator.SAME_POSE_LOWER_THRESHOLD:
                break

            self.path_pointer += 1
        # determine the action progress
        action_progress = self.path_pointer / float(len(self.generated_path))

        # determine if this is the end of an action
        end_of_action = False
        if action_progress >= 0.9999:
            end_of_action = True
            action_progress = 1.

        new_path_plan = PathPlan()
        if end_of_action:
            self.reset()
        else:
            new_path_plan.tracking_pose = Pose2D()
            new_path_plan.tracking_pose.x = self.generated_path[self.path_pointer].x
            new_path_plan.tracking_pose.y = self.generated_path[self.path_pointer].y
            new_path_plan.tracking_pose.theta = self.generated_path[self.path_pointer].theta
            new_path_plan.tracking_speed = self.generated_path[self.path_pointer].speed
        new_path_plan.reset_sim = False
        new_path_plan.end_of_action = end_of_action
        new_path_plan.action_progress = action_progress
        new_path_plan.path_planner_terminate = end_of_action

        # future poses
        path_pointer = self.path_pointer
        new_path_plan.future_poses = []
        while path_pointer < len(self.generated_path):
            new_pose = Pose2D()
            new_pose.x = self.generated_path[self.path_pointer].x
            new_pose.y = self.generated_path[self.path_pointer].y
            new_pose.theta = self.generated_path[self.path_pointer].theta
            new_path_plan.future_poses.append(new_pose)
            path_pointer += 1

        return new_path_plan
