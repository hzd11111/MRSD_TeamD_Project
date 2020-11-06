from options import Scenario, TrafficLightStatus, PedestrainPriority, StopLineStatus
from utility import EnvDesc
import numpy as np
import itertools
import sys
sys.path.append("../../carla_utils/utils")

# ROS Packages
# other packages


class StateManager:
    """
    A class to encompass all the logic needed to create state embeddings
    """

    def __init__(self):
        pass

    def appendVehicleState(self, env_state, vehicle_state):
        env_state.append(vehicle_state.vehicle_location.x)
        env_state.append(vehicle_state.vehicle_location.y)
        env_state.append(vehicle_state.vehicle_location.theta)
        env_state.append(vehicle_state.vehicle_speed)
        return

    def selectLane(self, lane, left=True):
        """
        Select the right or left lane lane
        """
        if (left):
            return lane.left_to_the_current and lane.adjacent_lane
        else:
            return (not lane.left_to_the_current) and lane.adjacent_lane

    def createTrafficLightOneHotVec(self, light, pedestrian=False):
        """
        Create a light to one-hot-vec based on type of light
        """
        if pedestrian:
            if light is PedestrainPriority.GREEN:
                return [0, 0, 1]
            elif light is PedestrainPriority.RED:
                return [0, 1, 0]
        else:
            if light == TrafficLightStatus.GREEN:
                return [1, 0, 0]
            elif light == TrafficLightStatus.YELLOW:
                return [0, 1, 0]
            else:
                return [0, 0, 1]

    def createStopLineToOneHotVec(self, stop_line):
        """
        Create a one hot vec for stop line status
        """
        if stop_line is StopLineStatus.NO_STOP:
            return [1, 0, 0, 0]
        elif stop_line is StopLineStatus.STRAIGHT_STOP:
            return [0, 1, 0, 0]
        elif stop_line is StopLineStatus.LEFT_TURN_STOP:
            return [0, 0, 1, 0]
        elif stop_line is StopLineStatus.RIGHT_TURN_STOP:
            return [0, 0, 0, 1]

    def createLaneChangeState(self, env_desc, left=True):
        """
        Create a dictionary of states. They keys are "adjacent", "front", "back", "pedestrians",
        "lane_distance". Each value for the actor types is a numpy array of the size (n X 6),
        where is the number of actors of that type
        """
        # extract all vehicles from adjacent lanes
        # TODO: Should I choose vehicles only from the lane to which we are shifting or all adjacent lanes?
        lane_distance = 0
        dummy_vehicle = [1000, 1000, np.cos(0), np.sin(0), 0, 0, 0]
        pedestrian_states = []
        adj_lane_vehicles_states = []
        ego_vehicle_state = [
            env_desc.cur_vehicle_state.location_frenet.x,
            env_desc.cur_vehicle_state.location_frenet.y,
            np.cos(env_desc.cur_vehicle_state.location_frenet.theta),
            np.sin(env_desc.cur_vehicle_state.location_frenet.theta),
            env_desc.cur_vehicle_state.speed,
            env_desc.cur_vehicle_state.acceleration]
        adj_lane_vehs = []

        for lane in env_desc.adjacent_lanes:
            if self.selectLane(lane, left):
                # Add relevant vehicles
                for vehicle in lane.lane_vehicles:
                    adj_lane_vehs.append([vehicle, lane])
                    
                # Add relevant pedestrians
                for pedestrian in lane.crossing_pedestrain:
                    pedestrian_in_ego = pedestrian.fromControllingVehicle(
                        env_desc.cur_vehicle_state.location_frenet,
                        lane)
                    pedestrian_state = [pedestrian_in_ego.location_frenet.x,
                                        pedestrian_in_ego.location_frenet.y,
                                        np.cos(pedestrian_in_ego.location_frenet.theta),
                                        np.sin(pedestrian_in_ego.location_frenet.theta),
                                        pedestrian.speed,
                                        pedestrian.acceleration,
                                        1]
                    pedestrian_states.append(pedestrian_state)
                # Get the lane distance
                lane_distance = lane.lane_distance
                break

        if len(adj_lane_vehs) > 5:
            adj_lane_vehs = sorted(
                lambda item: item[0].location_global.distance(env_desc.cur_vehicle_state.location_global),
                adj_lane_vehs)[:5]

        for vehicle, lane in adj_lane_vehs:
            vehicle_in_ego = vehicle.fromControllingVehicle(
                env_desc.cur_vehicle_state.location_frenet,
                lane)
            vehicle_state = [vehicle_in_ego.x,
                             vehicle_in_ego.y,
                             np.cos(vehicle_in_ego.theta),
                             np.sin(vehicle_in_ego.theta),
                             vehicle.speed,
                             vehicle.acceleration,
                             1]
            adj_lane_vehicles_states.append(vehicle_state)

        if len(adj_lane_vehicles_states) < 5:
            dummy_vehicles = list(itertools.repeat(
                dummy_vehicle, 5 - len(adj_lane_vehicles_states)))
            adj_lane_vehicles_states += dummy_vehicles

        # extract front vehicle and back vehicle and pedestrian in current lane
        # TODO: Make sure vehicle in front or back are always present
        current_lane = env_desc.current_lane
        front_vehicle = current_lane.VehicleInFront(env_desc.cur_vehicle_state)
        back_vehicle = current_lane.VehicleBehind(env_desc.cur_vehicle_state)

        for pedestrian in current_lane.crossing_pedestrain:
            pedestrian_in_ego = pedestrian.fromControllingVehicle(
                env_desc.cur_vehicle_state.location_frenet,
                env_desc.current_lane)
            pedestrian_state = [pedestrian_in_ego.location_frenet.x,
                                pedestrian_in_ego.location_frenet.y,
                                np.cos(pedestrian_in_ego.location_frenet.theta),
                                np.sin(pedestrian_in_ego.location_frenet.theta),
                                pedestrian.speed,
                                pedestrian.acceleration,
                                1]
            pedestrian_states.append(pedestrian_state)

        if len(pedestrian_states) > 3:
            pedestrian_states = sorted(pedestrian_states, lambda item: item[0])[:3]
        elif len(pedestrian_states) < 3:
            dummy_vehicles = list(itertools.repeat(
                dummy_vehicle, 3 - len(pedestrian_states)))
            pedestrian_states += dummy_vehicles
        # TODO: Should I consider when pedestrians are more than 5? How to sort?
        # TODO: Should I use global speed after converting to frenet?
        if front_vehicle is None:
            front_vehicle_state = dummy_vehicle
        else:
            front_vehicle_state = [front_vehicle.location_frenet.x,
                                   front_vehicle.location_frenet.y,
                                   np.cos(front_vehicle.location_frenet.theta),
                                   np.sin(front_vehicle.location_frenet.theta),
                                   front_vehicle.speed,
                                   front_vehicle.acceleration,
                                   1]

        if back_vehicle is None:
            back_vehicle_state = dummy_vehicle
        else:
            back_vehicle_state = [back_vehicle.location_frenet.x,
                                  back_vehicle.location_frenet.y,
                                  np.cos(back_vehicle.location_frenet.theta),
                                  np.sin(back_vehicle.location_frenet.theta),
                                  back_vehicle.speed,
                                  back_vehicle.acceleration,
                                  1]

        distance_to_stop_line = env_desc.cur_vehicle_state.traffic_light_stop_distance
        print("Distance to stop line is",distance_to_stop_line)
        # concatenate all the states and lane distance
        entire_state = ego_vehicle_state + \
            [coord for state in adj_lane_vehicles_states for coord in state] + \
            front_vehicle_state + \
            back_vehicle_state + \
            [coord for state in pedestrian_states for coord in state] + \
            [lane_distance, distance_to_stop_line]
        try:
            assert(len(entire_state) == 78)
        except e:
            import ipdb; ipdb.set_trace()
        return np.array(entire_state)

    def createLaneFollowingState(self, env_desc):
        """
        Creates a lane following state with front vehicle,
        back vehicle and pedestrians
        """
        dummy_vehicle = [1000, 1000, np.cos(0), np.sin(0), 0, 0, 0]
        pedestrian_states = []
        ego_vehicle_state = [
            env_desc.cur_vehicle_state.location_frenet.x,
            env_desc.cur_vehicle_state.location_frenet.y,
            np.cos(env_desc.cur_vehicle_state.location_frenet.theta),
            np.sin(env_desc.cur_vehicle_state.location_frenet.theta),
            env_desc.cur_vehicle_state.speed,
            env_desc.cur_vehicle_state.acceleration]

        # extract front vehicle and back vehicle and pedestrian in current lane
        # TODO: Make sure vehicle in front or back are always present
        current_lane = env_desc.current_lane
        front_vehicle = current_lane.VehicleInFront(env_desc.cur_vehicle_state)
        back_vehicle = current_lane.VehicleBehind(env_desc.cur_vehicle_state)

        for pedestrian in current_lane.crossing_pedestrain:
            pedestrian_in_ego = pedestrian.fromControllingVehicle(
                env_desc.cur_vehicle_state.location_frenet,
                env_desc.current_lane)
            pedestrian_state = [pedestrian_in_ego.location_frenet.x,
                                pedestrian_in_ego.location_frenet.y,
                                np.cos(pedestrian_in_ego.location_frenet.theta),
                                np.sin(pedestrian_in_ego.location_frenet.theta),
                                pedestrian.speed,
                                pedestrian.acceleration,
                                1]
            pedestrian_states.append(pedestrian_state)

        if len(pedestrian_states) > 3:
            pedestrian_states = sorted(pedestrian_states, lambda item: item[0])[:, 3]
        elif len(pedestrian_states) < 3:
            dummy_vehicles = list(itertools.repeat(
                dummy_vehicle, 3 - len(pedestrian_states)))
            pedestrian_states += dummy_vehicles

        if front_vehicle is None:
            front_vehicle_state = dummy_vehicle
        else:
            front_vehicle_state = [front_vehicle.location_frenet.x,
                                   front_vehicle.location_frenet.y,
                                   np.cos(front_vehicle.location_frenet.theta),
                                   np.sin(front_vehicle.location_frenet.theta),
                                   front_vehicle.speed,
                                   front_vehicle.acceleration,
                                   1]

        if back_vehicle is None:
            back_vehicle_state = dummy_vehicle
        else:
            back_vehicle_state = [back_vehicle.location_frenet.x,
                                  back_vehicle.location_frenet.y,
                                  np.cos(back_vehicle.location_frenet.theta),
                                  np.sin(back_vehicle.location_frenet.theta),
                                  back_vehicle.speed,
                                  back_vehicle.acceleration,
                                  1]

        # concatenate all the states and lane distance
        entire_state = ego_vehicle_state + \
            front_vehicle_state + \
            back_vehicle_state + \
            [coord for state in pedestrian_states for coord in state]

        assert(len(entire_state) == 41)
        return np.array(entire_state)

    def createIntersectionStraightState(self, env_desc):
        """
        Create a state for intersection negotiation state when turning
        """
        dummy_vehicle = [1000, 1000, np.cos(0), np.sin(0), 0, 0] + [0, 0, 1] + [0]
        dummy_ped = [1000, 1000, np.cos(0), np.sin(0), 0, 0] + [0, 1] + [0]
        pedestrian_states = []
        ego_vehicle_state = [
            env_desc.cur_vehicle_state.location_frenet.x,
            env_desc.cur_vehicle_state.location_frenet.y,
            np.cos(env_desc.cur_vehicle_state.location_frenet.theta),
            np.sin(env_desc.cur_vehicle_state.location_frenet.theta),
            env_desc.cur_vehicle_state.speed,
            env_desc.cur_vehicle_state.acceleration] + \
            self.createTrafficLightOneHotVec(env_desc.cur_vehicle_state.traffic_light_status)

        # extract front vehicle and back vehicle and pedestrian in current lane
        # TODO: Make sure vehicle in front or back are always present
        current_lane = env_desc.current_lane
        front_vehicle = current_lane.VehicleInFront(env_desc.cur_vehicle_state)
        back_vehicle = current_lane.VehicleBehind(env_desc.cur_vehicle_state)

        for pedestrian in current_lane.crossing_pedestrain:
            pedestrian_in_ego = pedestrian.fromControllingVehicle(
                env_desc.cur_vehicle_state.location_frenet,
                env_desc.current_lane)
            pedestrian_state = [pedestrian_in_ego.x,
                                pedestrian_in_ego.y,
                                np.cos(pedestrian_in_ego.theta),
                                np.sin(pedestrian_in_ego.theta),
                                pedestrian.speed,
                                pedestrian.acceleration] + \
                self.createTrafficLightOneHotVec(pedestrian.priority_status, True) + [1]
            pedestrian_states.append(pedestrian_state)

        if len(pedestrian_states) > 3:
            pedestrian_states = sorted(pedestrian_states, lambda item: item[0])[:,3]
        elif len(pedestrian_states) < 3:
            dummy_peds = list(itertools.repeat(
                dummy_ped, 3 - len(pedestrian_states)))
            pedestrian_states += dummy_peds

        if front_vehicle is None:
            front_vehicle_state = dummy_vehicle
        else:
            front_vehicle_state = [front_vehicle.location_frenet.x,
                                   front_vehicle.location_frenet.y,
                                   np.cos(front_vehicle.location_frenet.theta),
                                   np.sin(front_vehicle.location_frenet.theta),
                                   front_vehicle.speed,
                                   front_vehicle.acceleration] + \
                self.createTrafficLightOneHotVec(front_vehicle.traffic_light_status) + [1]

        if back_vehicle is None:
            back_vehicle_state = dummy_vehicle
        else:
            back_vehicle_state = [back_vehicle.location_frenet.x,
                                  back_vehicle.location_frenet.y,
                                  np.cos(back_vehicle.location_frenet.theta),
                                  np.sin(back_vehicle.location_frenet.theta),
                                  back_vehicle.speed,
                                  back_vehicle.acceleration] + \
                self.createTrafficLightOneHotVec(back_vehicle.traffic_light_status) + [1]

        # identify vehicles from perpendicular lanes within 20m
        # TODO: How to select within 20m. Is it frenet x, or global distance
        # identify the vehicles from left turning opposite lane
        opposite_left_turning_lane_vehs = []
        perpendicular_lane_vehs = []
        perpendicular_lane_vehs_in_ego = []
        for lane in env_desc.next_intersection:
            for vehicle in lane.lane_vehicles:
                perpendicular_lane_vehs.append([vehicle, lane])

        for lane in env_desc.adjacent_lanes:
            if lane.same_direction is False and lane.left_turning_lane is True and lane.is_misc is False:
                for vehicle in lane.lane_vehicles:
                    opposite_left_turning_lane_vehs.append([vehicle, lane])

        # take 10 vehicles with the smalled abs frenet x from perpendicular_lane_vehs
        for vehicle, lane in perpendicular_lane_vehs:
            veh_in_ego = vehicle.fromControllingVehicle(env_desc.cur_vehicle_state.location_frenet, lane)
            perpendicular_lane_vehs_in_ego.append([veh_in_ego.x,
                                                   veh_in_ego.y,
                                                   np.cos(veh_in_ego.theta),
                                                   np.sin(veh_in_ego.theta),
                                                   vehicle.speed,
                                                   vehicle.acceleration] +
                                                  self.createTrafficLightOneHotVec(
                                                      vehicle.traffic_light_status) + [1])

        if len(perpendicular_lane_vehs_in_ego) > 10:
            perpendicular_lane_vehs_in_ego = sorted(
                perpendicular_lane_vehs_in_ego, key=lambda item: abs(item[0]))
            perpendicular_lane_vehs_in_ego = perpendicular_lane_vehs_in_ego[:10]
        else:
            dummy_vehicles = list(itertools.repeat(
                dummy_vehicle, 10 - len(perpendicular_lane_vehs_in_ego)))
            perpendicular_lane_vehs_in_ego += dummy_vehicles

        # take 5 closest in the opposite_left_turning_lane_vehs
        opposite_left_turning_lane_vehs_in_ego = []
        # if there are more than 5 vehicles than sort them globally and select 5 closest
        if len(opposite_left_turning_lane_vehs) > 5:
            opposite_left_turning_lane_vehs = sorted(
                opposite_left_turning_lane_vehs,
                key=lambda item: item[0].location_global.distance(
                    env_desc.cur_vehicle_state.location_global))
            opposite_left_turning_lane_vehs = opposite_left_turning_lane_vehs[:5]

        # collect the frenet coordinates
        for vehicle, lane in opposite_left_turning_lane_vehs:
            veh_in_ego = vehicle.fromControllingVehicle(env_desc.cur_vehicle_state.location_frenet, lane)
            opposite_left_turning_lane_vehs_in_ego.append([veh_in_ego.x,
                                                           veh_in_ego.y,
                                                           np.cos(veh_in_ego.theta),
                                                           np.sin(veh_in_ego.theta),
                                                           vehicle.speed,
                                                           vehicle.acceleration] +
                                                          self.createTrafficLightOneHotVec(
                                                              vehicle.traffic_light_status) + [1])

        # if the number of vehicles in the opposite left_turning_lane are less than 5 add dummy
        if len(opposite_left_turning_lane_vehs_in_ego) < 5:
            dummy_vehicles = list(itertools.repeat(
                dummy_vehicle, 5 - len(opposite_left_turning_lane_vehs_in_ego)))
            opposite_left_turning_lane_vehs_in_ego += dummy_vehicles

        # get vehicles in the intersection
        intersection_vehs_in_ego = []
        intersection_vehs = []
        for lane in env_desc.adjacent_lanes:
            if lane.is_misc is True:
                for vehicle in lane.lane_vehicles:
                    if vehicle.location_global.distance(
                        env_desc.cur_vehicle_state.location_global) < 40.0:
                        intersection_vehs.append(vehicle)

        # if number of vehicles in intersection are more then 3 then take 3 closest
        if len(intersection_vehs) > 3:
            intersection_vehs = sorted(
                intersection_vehs,
                key=lambda item: item.location_global.distance(
                    env_desc.cur_vehicle_state.location_global))
            intersection_vehs = intersection_vehs[:3]

        for vehicle in intersection_vehs:
            veh_in_ego = vehicle.fromControllingVehicle(env_desc.cur_vehicle_state.location_frenet,
                                                        env_desc.current_lane)
            intersection_vehs_in_ego.append([veh_in_ego.x,
                                             veh_in_ego.y,
                                             np.cos(veh_in_ego.theta),
                                             np.sin(veh_in_ego.theta),
                                             vehicle.speed,
                                             vehicle.acceleration] +
                                            self.createTrafficLightOneHotVec(
                                                vehicle.traffic_light_status) + [1])

        # if number of vehicles in intersection are less than 3 then add dummies
        if len(intersection_vehs_in_ego) < 3:
            dummy_vehicles = list(itertools.repeat(
                dummy_vehicle, 3 - len(intersection_vehs_in_ego)))
            intersection_vehs_in_ego += dummy_vehicles

        # Add the stop line status and segment start distance for current lane
        current_lane_status = []
        current_lane_status += self.createTrafficLightOneHotVec(
            env_desc.cur_vehicle_state.traffic_light_status)

        # new segment distance
        for point in env_desc.current_lane.lane_points:
            if point.lane_start is True:
                current_lane_status += [abs(point.frenet_pose.x)]
                break

        # for point in env_desc.current_lane.lane_points:
        #     if point.stop_line is not StopLineStatus.NO_STOP:
        #         current_lane_status += [(point.frenet_pose.x)]
        #         break

        distance_to_stop_line = env_desc.cur_vehicle_state.traffic_light_stop_distance
        current_lane_status += [distance_to_stop_line]
        print("Distance to stop line is", distance_to_stop_line)

        if len(current_lane_status) != 8:
            current_lane_status += list(itertools.repeat(
                0, 8 - len(current_lane_status)))

        
        # concatenate all the states and lane distance
        entire_state = current_lane_status + ego_vehicle_state + \
            front_vehicle_state + \
            back_vehicle_state + \
            [coord for state in pedestrian_states for coord in state] + \
            [coord for state in perpendicular_lane_vehs_in_ego for coord in state] + \
            [coord for state in opposite_left_turning_lane_vehs_in_ego for coord in state] +\
            [coord for state in intersection_vehs_in_ego for coord in state]

        print(len(entire_state))
        print(len(intersection_vehs_in_ego))
        assert(len(entire_state) == 244)
        return np.array(entire_state)

    def createIntersectionLeftTurnState(self, env_desc):
        """
        Create a state for intersection when taking a left turn
        """
        dummy_vehicle = [1000, 1000, np.cos(0), np.sin(0), 0, 0] + [0, 0, 1] + [0]
        dummy_ped = [1000, 1000, np.cos(0), np.sin(0), 0, 0] + [0, 1] + [0]
        pedestrian_states = []
        ego_vehicle_state = [
            env_desc.cur_vehicle_state.location_frenet.x,
            env_desc.cur_vehicle_state.location_frenet.y,
            np.cos(env_desc.cur_vehicle_state.location_frenet.theta),
            np.sin(env_desc.cur_vehicle_state.location_frenet.theta),
            env_desc.cur_vehicle_state.speed,
            env_desc.cur_vehicle_state.acceleration] + \
            self.createTrafficLightOneHotVec(env_desc.cur_vehicle_state.traffic_light_status)
        print(self.createTrafficLightOneHotVec(env_desc.cur_vehicle_state.traffic_light_status))
        # extract front vehicle and back vehicle and pedestrian in current lane
        # TODO: Make sure vehicle in front or back are always present
        current_lane = env_desc.current_lane
        front_vehicle = current_lane.VehicleInFront(env_desc.cur_vehicle_state)
        back_vehicle = current_lane.VehicleBehind(env_desc.cur_vehicle_state)

        # select pedestrians from all perpendicular lanes with directed_right = false
        for lane in env_desc.next_intersection:
            if lane.directed_right is False:
                for pedestrian in lane.crossing_pedestrain:
                    pedestrian_in_ego = pedestrian.fromControllingVehicle(
                        env_desc.cur_vehicle_state.location_frenet,
                        lane)
                    pedestrian_state = [pedestrian_in_ego.x,
                                        pedestrian_in_ego.y,
                                        np.cos(pedestrian_in_ego.theta),
                                        np.sin(pedestrian_in_ego.theta),
                                        pedestrian.speed,
                                        pedestrian.acceleration] + \
                        self.createTrafficLightOneHotVec(pedestrian.priority_status, True) + [1]
                    pedestrian_states.append(pedestrian_state)

        # filter out pedestrians with negative x frenet
        pedestrian_states = list(filter(lambda item: item[0] >= 0, pedestrian_states))
        # select top 3 or add dummy pedestrians
        if len(pedestrian_states) > 3:
            pedestrian_states = sorted(pedestrian_states, lambda item: item[1])[:3]
        elif len(pedestrian_states) < 3:
            dummy_peds = list(itertools.repeat(
                dummy_ped, 3 - len(pedestrian_states)))
            pedestrian_states += dummy_peds

        if front_vehicle is None:
            front_vehicle_state = dummy_vehicle
        else:
            front_vehicle_state = [front_vehicle.location_frenet.x,
                                   front_vehicle.location_frenet.y,
                                   np.cos(front_vehicle.location_frenet.theta),
                                   np.sin(front_vehicle.location_frenet.theta),
                                   front_vehicle.speed,
                                   front_vehicle.acceleration] + \
                self.createTrafficLightOneHotVec(front_vehicle.traffic_light_status) + [1]

        if back_vehicle is None:
            back_vehicle_state = dummy_vehicle
        else:
            back_vehicle_state = [back_vehicle.location_frenet.x,
                                  back_vehicle.location_frenet.y,
                                  np.cos(back_vehicle.location_frenet.theta),
                                  np.sin(back_vehicle.location_frenet.theta),
                                  back_vehicle.speed,
                                  back_vehicle.acceleration] + \
                self.createTrafficLightOneHotVec(back_vehicle.traffic_light_status) + [1]
        # identify vehicles from perpendicular lanes within 20m
        # TODO: How to select within 20m. Is it frenet x, or global distance
        parallel_lane_vehs = []
        perpendicular_lane_vehs = []
        for lane in env_desc.next_intersection:
            for vehicle in lane.lane_vehicles:
                perpendicular_lane_vehs.append([vehicle, lane])

        # identify the vehicles from parallel lane
        for lane in env_desc.adjacent_lanes:
            if lane.same_direction is False and lane.is_misc is False:
                for vehicle in lane.lane_vehicles:
                    parallel_lane_vehs.append([vehicle, lane])

        # take 10 vehicles with the smalled abs frenet x from perpendicular_lane_vehs
        perpendicular_lane_vehs_in_ego = []
        for vehicle, lane in perpendicular_lane_vehs:
            veh_in_ego = vehicle.fromControllingVehicle(env_desc.cur_vehicle_state.location_frenet, lane)
            perpendicular_lane_vehs_in_ego.append([veh_in_ego.x,
                                                   veh_in_ego.y,
                                                   np.cos(veh_in_ego.theta),
                                                   np.sin(veh_in_ego.theta),
                                                   vehicle.speed,
                                                   vehicle.acceleration] +
                                                  self.createTrafficLightOneHotVec(
                                                      vehicle.traffic_light_status) + [1])

        if len(perpendicular_lane_vehs_in_ego) > 10:
            perpendicular_lane_vehs_in_ego = sorted(
                perpendicular_lane_vehs_in_ego, key=lambda item: abs(item[0]))
            perpendicular_lane_vehs_in_ego = perpendicular_lane_vehs_in_ego[:10]
        elif len(perpendicular_lane_vehs_in_ego) < 10:
            dummy_vehicles = list(itertools.repeat(
                dummy_vehicle, 10 - len(perpendicular_lane_vehs_in_ego)))
            perpendicular_lane_vehs_in_ego += dummy_vehicles

        # take 10 closest in the parallel lane
        parallel_lane_vehs_in_ego = []
        # if there are more than 10 vehicles than sort them globally and select 10 closest
        # collect the frenet coordinates
        for vehicle, lane in parallel_lane_vehs:
            veh_in_ego = vehicle.fromControllingVehicle(env_desc.cur_vehicle_state.location_frenet, lane)
            parallel_lane_vehs_in_ego.append([veh_in_ego.x,
                                              veh_in_ego.y,
                                              np.cos(veh_in_ego.theta),
                                              np.sin(veh_in_ego.theta),
                                              vehicle.speed,
                                              vehicle.acceleration] +
                                             self.createTrafficLightOneHotVec(
                                                 vehicle.traffic_light_status) + [1])

        # filter out vehicles with negative x frenet
        parallel_lane_vehs_in_ego = list(filter(lambda item: item[0] >= 0, parallel_lane_vehs_in_ego))
        # if the number of vehicles in the parallel lanes are less than 10 add dummy
        if len(parallel_lane_vehs_in_ego) > 10:
            parallel_lane_vehs_in_ego = parallel_lane_vehs_in_ego[:10]
        elif len(parallel_lane_vehs_in_ego) < 10:
            dummy_vehicles = list(itertools.repeat(
                dummy_vehicle, 10 - len(parallel_lane_vehs_in_ego)))
            parallel_lane_vehs_in_ego += dummy_vehicles

        intersection_vehs_in_ego = []
        intersection_vehs = []
        for lane in env_desc.adjacent_lanes:
            if lane.is_misc is True:
                for vehicle in lane.lane_vehicles:
                    if vehicle.location_global.distance(
                        env_desc.cur_vehicle_state.location_global) < 40.0:
                        intersection_vehs.append(vehicle)

        # if number of vehicles in intersection are more then 3 then take 3 closest
        if len(intersection_vehs) > 3:
            intersection_vehs = sorted(
                intersection_vehs,
                key=lambda item: item.location_global.distance(
                    env_desc.cur_vehicle_state.location_global))
            intersection_vehs = intersection_vehs[:3]

        for vehicle in intersection_vehs:
            veh_in_ego = vehicle.fromControllingVehicle(env_desc.cur_vehicle_state.location_frenet,
                                                        env_desc.current_lane)
            intersection_vehs_in_ego.append([veh_in_ego.x,
                                             veh_in_ego.y,
                                             np.cos(veh_in_ego.theta),
                                             np.sin(veh_in_ego.theta),
                                             vehicle.speed,
                                             vehicle.acceleration] +
                                            self.createTrafficLightOneHotVec(
                                                vehicle.traffic_light_status) + [1])

        # if number of vehicles in intersection are less than 3 then add dummies
        if len(intersection_vehs_in_ego) < 3:
            dummy_vehicles = list(itertools.repeat(
                dummy_vehicle, 3 - len(intersection_vehs_in_ego)))
            intersection_vehs_in_ego += dummy_vehicles

        # get the stopline distance, light status and merging lane distance
        current_lane_status = []
        current_lane_status += self.createTrafficLightOneHotVec(
            env_desc.cur_vehicle_state.traffic_light_status)

        # stop line distance
        # for point in env_desc.current_lane.lane_points:
        #     if point.stop_line is not StopLineStatus.NO_STOP:
        #         current_lane_status += [abs(point.frenet_pose.x)]
        
        distance_to_stop_line = env_desc.cur_vehicle_state.traffic_light_stop_distance
        current_lane_status += [distance_to_stop_line]
        print("Distance to stop line is", distance_to_stop_line)

        # get the merging distance
        min_merging_dist = 100
        for lane in env_desc.next_intersection:
            if lane.directed_right is False:
                if min_merging_dist > lane.intersecting_distance:
                    min_merging_dist = lane.intersecting_distance
        current_lane_status += [min_merging_dist]

        if len(current_lane_status) != 8:
            current_lane_status += list(itertools.repeat(
                0, 8 - len(current_lane_status)))

        # concatenate all the states and lane distance
        entire_state = current_lane_status + ego_vehicle_state + \
            front_vehicle_state + \
            back_vehicle_state + \
            [coord for state in pedestrian_states for coord in state] + \
            [coord for state in perpendicular_lane_vehs_in_ego for coord in state] + \
            [coord for state in parallel_lane_vehs_in_ego for coord in state] + \
            [coord for state in intersection_vehs_in_ego for coord in state]

        print(len(entire_state))
        assert(len(entire_state) == 294)  # TODO: Update this after adding lights
        return np.array(entire_state)

    def createIntersectionRightTurnState(self, env_desc):
        """
        Create a state for intersection when taking a right turn
        """
        dummy_vehicle = [1000, 1000, np.cos(0), np.sin(0), 0, 0] + [0, 0, 1] + [0]
        dummy_ped = [1000, 1000, np.cos(0), np.sin(0), 0, 0] + [0, 1] + [0]
        pedestrian_states_perp = []
        pedestrian_states_curr = []
        perpendicular_lane_vehs = []
        ego_vehicle_state = [
            env_desc.cur_vehicle_state.location_frenet.x,
            env_desc.cur_vehicle_state.location_frenet.y,
            np.cos(env_desc.cur_vehicle_state.location_frenet.theta),
            np.sin(env_desc.cur_vehicle_state.location_frenet.theta),
            env_desc.cur_vehicle_state.speed,
            env_desc.cur_vehicle_state.acceleration] + \
            self.createTrafficLightOneHotVec(env_desc.cur_vehicle_state.traffic_light_status)

        # extract front vehicle and back vehicle and pedestrian in current lane
        # TODO: Make sure vehicle in front or back are always present
        current_lane = env_desc.current_lane
        front_vehicle = current_lane.VehicleInFront(env_desc.cur_vehicle_state)
        back_vehicle = current_lane.VehicleBehind(env_desc.cur_vehicle_state)

        # select pedestrians from all perpendicular lanes which we turn into
        # TODO: Fix this after right_most lane is added to message lane.right_most_lane
        for lane in env_desc.next_intersection:
            if True is True and lane.directed_right is True:
                for pedestrian in lane.crossing_pedestrain:
                    pedestrian_in_ego = pedestrian.fromControllingVehicle(
                        env_desc.cur_vehicle_state.location_frenet,
                        lane)
                    pedestrian_state = [pedestrian_in_ego.x,
                                        pedestrian_in_ego.y,
                                        np.cos(pedestrian_in_ego.theta),
                                        np.sin(pedestrian_in_ego.theta),
                                        pedestrian.speed,
                                        pedestrian.acceleration] + \
                        self.createTrafficLightOneHotVec(pedestrian.priority_status, True) + [1]
                    pedestrian_states_perp.append(pedestrian_state)
                # identify vehicles from perpendicular lanes within 20m
                # TODO: How to select within 20m. Is it frenet x, or global distance
                for vehicle in lane.lane_vehicles:
                    perpendicular_lane_vehs.append([vehicle, lane])

        # select smallest 3 y frenet pedestrians or add dummy pedestrians
        # TODO: Do i have to filter for positive x like in left turn?
        if len(pedestrian_states_perp) > 3:
            pedestrian_states_perp = sorted(pedestrian_states_perp, lambda item: item[1])[:3]
        elif len(pedestrian_states_perp) < 3:
            dummy_peds = list(itertools.repeat(
                dummy_ped, 3 - len(pedestrian_states_perp)))
            pedestrian_states_perp += dummy_peds

        # select pedestrians from current_lane
        for pedestrian in env_desc.current_lane.crossing_pedestrain:
            pedestrian_in_ego = pedestrian.fromControllingVehicle(
                env_desc.cur_vehicle_state.location_frenet,
                env_desc.current_lane)
            pedestrian_state = [pedestrian_in_ego.x,
                                pedestrian_in_ego.y,
                                np.cos(pedestrian_in_ego.theta),
                                np.sin(pedestrian_in_ego.theta),
                                pedestrian.speed,
                                pedestrian.acceleration] + \
                self.createTrafficLightOneHotVec(pedestrian.priority_status, True) + [1]
            pedestrian_states_curr.append(pedestrian_state)

        # select smallest 3 y frenet pedestrians or add dummy pedestrians
        # TODO: Do i have to filter for positive x like in left turn?
        if len(pedestrian_states_curr) > 3:
            pedestrian_states_curr = sorted(pedestrian_states_curr, lambda item: item[1])[:3]
        elif len(pedestrian_states_curr) < 3:
            dummy_peds = list(itertools.repeat(
                dummy_ped, 3 - len(pedestrian_states_curr)))
            pedestrian_states_curr += dummy_peds

        if front_vehicle is None:
            front_vehicle_state = dummy_vehicle
        else:
            front_vehicle_state = [front_vehicle.location_frenet.x,
                                   front_vehicle.location_frenet.y,
                                   np.cos(front_vehicle.location_frenet.theta),
                                   np.sin(front_vehicle.location_frenet.theta),
                                   front_vehicle.speed,
                                   front_vehicle.acceleration] + \
                self.createTrafficLightOneHotVec(front_vehicle.traffic_light_status) + [1]

        if back_vehicle is None:
            back_vehicle_state = dummy_vehicle
        else:
            back_vehicle_state = [back_vehicle.location_frenet.x,
                                  back_vehicle.location_frenet.y,
                                  np.cos(back_vehicle.location_frenet.theta),
                                  np.sin(back_vehicle.location_frenet.theta),
                                  back_vehicle.speed,
                                  back_vehicle.acceleration] + \
                self.createTrafficLightOneHotVec(back_vehicle.traffic_light_status) + [1]

        # take 5 vehicles with the smalled abs frenet x from perpendicular_lane_vehs
        perpendicular_lane_vehs_in_ego = []
        for vehicle, lane in perpendicular_lane_vehs:
            veh_in_ego = vehicle.fromControllingVehicle(env_desc.cur_vehicle_state.location_frenet,
                                                        lane)
            perpendicular_lane_vehs_in_ego.append([veh_in_ego.x,
                                                   veh_in_ego.y,
                                                   np.cos(veh_in_ego.theta),
                                                   np.sin(veh_in_ego.theta),
                                                   vehicle.speed,
                                                   vehicle.acceleration] +
                                                  self.createTrafficLightOneHotVec(
                                                      vehicle.traffic_light_status) + [1])

        if len(perpendicular_lane_vehs_in_ego) > 5:
            perpendicular_lane_vehs_in_ego = sorted(
                perpendicular_lane_vehs_in_ego, key=lambda item: abs(item[0]))
            perpendicular_lane_vehs_in_ego = perpendicular_lane_vehs_in_ego[:5]
        elif len(perpendicular_lane_vehs_in_ego) < 5:
            dummy_vehicles = list(itertools.repeat(
                dummy_vehicle, 5 - len(perpendicular_lane_vehs_in_ego)))
            perpendicular_lane_vehs_in_ego += dummy_vehicles

        intersection_vehs_in_ego = []
        intersection_vehs = []
        for lane in env_desc.adjacent_lanes:
            if lane.is_misc is True:
                for vehicle in lane.lane_vehicles:
                    if vehicle.location_global.distance(
                        env_desc.cur_vehicle_state.location_global) < 40.0:
                        intersection_vehs.append(vehicle)

        # if number of vehicles in intersection are more then 3 then take 3 closest
        if len(intersection_vehs) > 3:
            intersection_vehs = sorted(
                intersection_vehs,
                key=lambda item: item.location_global.distance(
                    env_desc.cur_vehicle_state.location_global))
            intersection_vehs = intersection_vehs[:3]

        for vehicle in intersection_vehs:
            veh_in_ego = vehicle.fromControllingVehicle(env_desc.cur_vehicle_state.location_frenet,
                                                        env_desc.current_lane)
            intersection_vehs_in_ego.append([veh_in_ego.x,
                                             veh_in_ego.y,
                                             np.cos(veh_in_ego.theta),
                                             np.sin(veh_in_ego.theta),
                                             vehicle.speed,
                                             vehicle.acceleration] +
                                            self.createTrafficLightOneHotVec(
                                                vehicle.traffic_light_status) + [1])

        # if number of vehicles in intersection are less than 3 then add dummies
        if len(intersection_vehs_in_ego) < 3:
            dummy_vehicles = list(itertools.repeat(
                dummy_vehicle, 3 - len(intersection_vehs_in_ego)))
            intersection_vehs_in_ego += dummy_vehicles

        # get the stopline distance, light status and merging lane distance
        current_lane_status = []

        current_lane_status += self.createTrafficLightOneHotVec(
            env_desc.cur_vehicle_state.traffic_light_status)

        # stop line distance
        # for point in env_desc.current_lane.lane_points:
        #     if point.stop_line is not StopLineStatus.NO_STOP:
        #         current_lane_status += [(point.frenet_pose.x)]
        #         break

        distance_to_stop_line = env_desc.cur_vehicle_state.traffic_light_stop_distance
        current_lane_status += [distance_to_stop_line]
        print("Distance to stop line is", distance_to_stop_line)

        # get the merging distance
        min_merging_dist = 100
        for lane in env_desc.next_intersection:
            if lane.directed_right is True:
                if min_merging_dist > lane.intersecting_distance:
                    min_merging_dist = lane.intersecting_distance
        current_lane_status += [min_merging_dist]

        if len(current_lane_status) != 8:
            current_lane_status += list(itertools.repeat(
                0, 8 - len(current_lane_status)))

        # concatenate all the states and lane distance
        entire_state = current_lane_status + ego_vehicle_state + \
            front_vehicle_state + \
            back_vehicle_state + \
            [coord for state in pedestrian_states_curr for coord in state] + \
            [coord for state in pedestrian_states_perp for coord in state] + \
            [coord for state in perpendicular_lane_vehs_in_ego for coord in state] + \
            [coord for state in intersection_vehs_in_ego for coord in state]

        print(len(entire_state))
        assert(len(entire_state) == 171)  # TODO: Update this after adding lights
        return np.array(entire_state)

    def embedState(self, env_desc: EnvDesc, scenario: Scenario):
        """
        Create a state embedding vector from message recieved
        Args:
        :param env_desc: (EnvironmentState) a ROS message describing the environment
        """
        # based on scenario create the embedding accordingly
        if scenario == Scenario.SWITCH_LANE_LEFT:
            return self.createLaneChangeState(env_desc, left=True)
        elif scenario == Scenario.SWITCH_LANE_RIGHT:
            return self.createLaneChangeState(env_desc, left=False)
        elif scenario == Scenario.LANE_FOLLOWING:
            return self.createLaneFollowingState(env_desc)
        elif scenario == Scenario.GO_STRAIGHT:
            return self.createIntersectionStraightState(env_desc)
        elif scenario == Scenario.LEFT_TURN:
            return self.createIntersectionLeftTurnState(env_desc)
        elif scenario == Scenario.RIGHT_TURN:
            return self.createIntersectionRightTurnState(env_desc)
        else:
            return {}
