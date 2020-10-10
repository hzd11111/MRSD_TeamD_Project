#!/usr/bin/env python
import rospy

from shapely.geometry import LineString, Point

from geometry_msgs.msg import Pose2D as Pose2DMsg
from carla_utils.msg import PathPlanMsg
from carla_utils.msg import GlobalPathPointMsg
from carla_utils.msg import GlobalPathMsg
from carla_utils.msg import LaneStatusMsg
from carla_utils.msg import CurrentLaneMsg
from carla_utils.msg import ParallelLaneMsg
from carla_utils.msg import PerpendicularLaneMsg
from carla_utils.msg import VehicleMsg
from carla_utils.msg import RewardInfoMsg
from carla_utils.msg import EnvDescMsg
from carla_utils.msg import LanePointMsg


from actors import *
from options import *
from functional_utility import Pose2D, Frenet

sys.path.append("../../carla_bridge/scripts/cartesian_to_frenet")

from cartesian_to_frenet import (
    get_cartesian_from_frenet,
    get_frenet_from_cartesian,
    get_path_linestring,
)


class PathPlan(object):
    __slots__ = [
        "tracking_pose",
        "future_poses",
        "tracking_speed",
        "reset_sim",
        "path_planner_terminate",
        "end_of_action",
        "action_progress",
    ]

    def __init__(
        self,
        tracking_pose=Pose2D(),
        future_poses=[],
        tracking_speed=0.0,
        reset_sim=False,
        path_planner_terminate=False,
        end_of_action=False,
        action_progress=0.0,
    ):
        self.tracking_pose = tracking_pose
        self.future_poses = future_poses
        self.tracking_speed = tracking_speed
        self.reset_sim = reset_sim
        self.path_planner_terminate = path_planner_terminate
        self.end_of_action = end_of_action
        self.action_progress = action_progress

    @classmethod
    def fromRosMsg(cls, path_plan_msg):
        obj = cls.__new__(cls)
        obj.tracking_pose = Pose2D.fromRosMsg(path_plan_msg.tracking_pose)
        obj.future_poses = path_plan_msg.future_poses
        obj.tracking_speed = path_plan_msg.tracking_speed
        obj.reset_sim = path_plan_msg.reset_sim
        obj.path_planner_terminate = path_plan_msg.path_planner_terminate
        obj.end_of_action = path_plan_msg.end_of_action
        obj.action_progress = path_plan_msg.action_progress
        return obj

    def toRosMsg(self):
        msg = PathPlanMsg()
        msg.tracking_pose = self.tracking_pose.toRosMsg()
        msg.future_poses = self.future_poses
        msg.tracking_speed = self.tracking_speed
        msg.reset_sim = self.reset_sim
        msg.path_planner_terminate = self.path_planner_terminate
        msg.end_of_action = self.end_of_action
        msg.action_progress = self.action_progress
        return msg


class GlobalPathPoint(object):
    __slots__ = ["global_pose", "lane_id", "action"]

    def __init__(self, global_pose=Pose2D(), lane_id=0, action=0):
        self.global_pose = global_pose
        self.lane_id = lane_id
        self.action = action

    @classmethod
    def fromRosMsg(cls, msg):
        obj = cls.__new__(cls)
        obj.global_pose = Pose2D.fromRosMsg(msg.global_pose)
        obj.lane_id = msg.lane_id
        obj.action = msg.action
        return obj

    def toRosMsg(self):
        msg = GlobalPathPointMsg()
        msg.global_pose = self.global_pose.toRosMsg()
        msg.lane_id = self.lane_id
        msg.action = self.action
        return msg


class GlobalPath(object):
    __slots__ = ["path_points"]

    def __init__(self, path_points=[]):
        self.path_points = path_points

    @classmethod
    def fromRosMsg(cls, msg):
        obj = cls.__new__(cls)
        obj.path_points = [GlobalPathPoint.fromRosMsg(p) for p in msg.path_points]
        return obj

    def toRosMsg(self):
        msg = GlobalPathMsg()
        msg.path_points = [p.toRosMsg() for p in self.path_points]
        return msg


class LanePoint(object):
    __slots__ = ["global_pose", "frenet_pose", "stop_line", "lane_start"]

    def __init__(
        self,
        global_pose=Pose2D(),
        frenet_pose=Frenet(),
        stop_line=None,
        lane_start=False,
    ):
        self.global_pose = global_pose
        self.frenet_pose = frenet_pose
        self.stop_line = stop_line
        self.lane_start = lane_start
        if self.stop_line is None:
            self.stop_line = StopLineStatus.NO_STOP

    @classmethod
    def fromRosMsg(cls, msg):
        obj = cls.__new__(cls)
        obj.global_pose = Pose2D.fromRosMsg(msg.global_pose)
        obj.frenet_pose = Frenet.fromRosMsg(msg.frenet_pose)
        obj.stop_line = StopLineStatus(msg.stop_line)
        obj.lane_start = msg.lane_start
        return obj

    def toRosMsg(self):
        msg = LanePointMsg()
        msg.global_pose = self.global_pose.toRosMsg()
        msg.frenet_pose = self.frenet_pose.toRosMsg()
        msg.stop_line = self.stop_line.value
        msg.lane_start = self.lane_start
        return msg


# Base class impl for Lanes
class LaneStatus(object):
    __slots__ = [
        "lane_vehicles",
        "lane_points",
        "lane_id",
        "crossing_pedestrain",
        "origin_global_pose",
    ]

    def __init__(
        self,
        lane_vehicles=[],
        lane_points=[],
        lane_id=0,
        crossing_pedestrain=[],
        origin_global_pose=Pose2D(),
    ):
        self.lane_vehicles = lane_vehicles
        self.lane_points = lane_points
        self.lane_id = lane_id
        self.crossing_pedestrain = crossing_pedestrain
        self.origin_global_pose = origin_global_pose

    @classmethod
    def fromRosMsg(cls, msg):
        obj = cls.__new__(cls)
        obj.lane_vehicles = [Vehicle.fromRosMsg(v) for v in msg.lane_vehicles]
        obj.lane_points = [LanePoint.fromRosMsg(lp) for lp in msg.lane_points]
        obj.lane_id = msg.lane_id
        obj.crossing_pedestrain = [
            Pedestrian.fromRosMsg(ped) for ped in msg.crossing_pedestrain
        ]
        obj.origin_global_pose = Pose2D.fromRosMsg(msg.origin_global_pose)
        return obj

    def toRosMsg(self):
        msg = LaneStatusMsg()
        msg.lane_vehicles = [v.toRosMsg() for v in self.lane_vehicles]
        msg.lane_points = [l.toRosMsg() for l in self.lane_points]
        msg.lane_id = self.lane_id
        msg.crossing_pedestrain = [p.toRosMsg() for p in self.crossing_pedestrain]
        msg.origin_global_pose = self.origin_global_pose.toRosMsg()
        return msg

    # TODO: add definition (Mayank)
    def frenetToGlobal(self, pose):

        linestring = get_path_linestring(self.lane_points)
        global_pose = get_cartesian_from_frenet(
            linestring, [pose.x, pose.y], pose.theta
        )

        return Pose2D(x=global_pose[0], y=global_pose[1], theta=global_pose[2])

    # TODO: add definition (Mayank)
    def GlobalToFrenet(self, pose):

        linestring = get_path_linestring(self.lane_points)
        frenet_pose = get_frenet_from_cartesian(
            linestring, Point(pose.x, pose.y), pose.theta
        )

        return Frenet(x=frenet_pose[0], y=frenet_pose[1], theta=frenet_pose[2])


class CurrentLane(LaneStatus):
    def __init__(self, *args, **kwargs):
        super(CurrentLane, self).__init__(*args, **kwargs)

    @classmethod
    def fromRosMsg(cls, msg):
        obj = cls.__new__(cls)
        obj = super(CurrentLane, obj).fromRosMsg(msg.lane_status)
        return obj

    def toRosMsg(self):
        msg = CurrentLaneMsg()
        msg.lane_status = super(CurrentLane, self).toRosMsg()
        return msg

    # TODO: add definition
    def VehicleInFront(curret_vehicle):
        vehicle_dummy = Vehicle(
            actor_id=-1, speed=-1, location_frenet=Pose2D(1000, 1000, 0)
        )
        return vehicle_dummy

    # TODO: add definition
    def VehicleBehind(curret_vehicle):
        vehicle_dummy = Vehicle(
            actor_id=-1, speed=-1, location_frenet=Pose2D(1000, 1000, 0)
        )
        return vehicle_dummy


class ParallelLane(LaneStatus):
    def __init__(
        self,
        lane_vehicles=[],
        lane_points=[],
        lane_id=0,
        crossing_pedestrain=[],
        origin_global_pose=Pose2D(),
        same_direction=False,
        left_to_the_current=False,
        adjacent_lane=False,
        lane_distance=0.0,
    ):
        super(ParallelLane, self).__init__(
            lane_vehicles, lane_points, lane_id, crossing_pedestrain, origin_global_pose
        )
        self.same_direction = same_direction
        self.left_to_the_current = left_to_the_current
        self.adjacent_lane = adjacent_lane
        self.lane_distance = lane_distance

    @classmethod
    def fromRosMsg(cls, msg):
        obj = cls.__new__(cls)
        obj = super(ParallelLane, obj).fromRosMsg(msg.lane_status)
        obj.same_direction = msg.same_direction
        obj.left_to_the_current = msg.left_to_the_current
        obj.adjacent_lane = msg.adjacent_lane
        obj.lane_distance = msg.lane_distance
        return obj

    def toRosMsg(self):
        msg = ParallelLaneMsg()
        msg.lane_status = super(ParallelLane, self).toRosMsg()
        msg.same_direction = self.same_direction
        msg.left_to_the_current = self.left_to_the_current
        msg.adjacent_lane = self.adjacent_lane
        msg.lane_distance = self.lane_distance
        return msg


class PerpendicularLane(LaneStatus):
    def __init__(
        self,
        lane_vehicles=[],
        lane_points=[],
        lane_id=0,
        crossing_pedestrain=[],
        origin_global_pose=Pose2D(),
        intersecting_distance=0.0,
        directed_right=False,
    ):
        super(PerpendicularLane, self).__init__(
            lane_vehicles, lane_points, lane_id, crossing_pedestrain, origin_global_pose
        )
        self.intersecting_distance = intersecting_distance
        self.directed_right = directed_right

    @classmethod
    def fromRosMsg(cls, msg):
        obj = cls.__new__(cls)
        obj = super(PerpendicularLane, obj).fromRosMsg(msg.lane_status)
        obj.intersecting_distance = msg.intersecting_distance
        obj.directed_right = msg.directed_right
        return obj

    def toRosMsg(self):
        msg = PerpendicularLaneMsg()
        msg.lane_status = super(PerpendicularLane, self).toRosMsg()
        msg.intersecting_distance = self.intersecting_distance
        msg.directed_right = self.directed_right
        return msg


class RewardInfo(object):
    __slots__ = [
        "collision",
        "time_elapsed",
        "new_run",
        "end_of_action",
        "action_progress",
        "current_action",
        "path_planner_terminate",
    ]

    def __init__(
        self,
        collision=False,
        time_elapsed=0.0,
        new_run=False,
        end_of_action=False,
        action_progress=0.0,
        current_action=None,
        path_planner_terminate=False,
    ):
        self.collision = collision
        self.time_elapsed = time_elapsed
        self.new_run = new_run
        self.end_of_action = end_of_action
        self.action_progress = action_progress
        self.current_action = current_action
        self.path_planner_terminate = path_planner_terminate
        if self.current_action is None:
            self.current_action = RLDecision.NO_ACTION

    @classmethod
    def fromRosMsg(cls, msg):
        obj = cls.__new__(cls)
        obj.collision = msg.collision
        obj.time_elapsed = msg.time_elapsed
        obj.new_run = msg.new_run
        obj.end_of_action = msg.end_of_action
        obj.action_progress = msg.action_progress
        obj.current_action = RLDecision(msg.current_action)
        obj.path_planner_terminate = msg.path_planner_terminate
        return obj

    def toRosMsg(self):
        msg = RewardInfoMsg()
        msg.collision = self.collision
        msg.time_elapsed = self.time_elapsed
        msg.new_run = self.new_run
        msg.end_of_action = self.end_of_action
        msg.action_progress = self.action_progress
        msg.current_action = self.current_action.value
        msg.path_planner_terminate = self.path_planner_terminate
        return msg


class EnvDesc(object):
    __slots__ = [
        "cur_vehicle_state",
        "current_lane",
        "next_intersection",
        "adjacent_lanes",
        "speed_limit",
        "reward_info",
        "pedestrain",
        "global_path",
    ]

    def __init__(
        self,
        cur_vehicle_state=Vehicle(),
        current_lane=CurrentLane(),
        next_intersection=[],
        adjacent_lanes=[],
        speed_limit=0.0,
        reward_info=RewardInfo(),
        global_path=GlobalPath(),
    ):
        self.cur_vehicle_state = cur_vehicle_state
        self.current_lane = current_lane
        self.next_intersection = next_intersection
        self.adjacent_lanes = adjacent_lanes
        self.speed_limit = speed_limit
        self.reward_info = reward_info
        self.global_path = global_path

    @classmethod
    def fromRosMsg(cls, msg):
        obj = cls.__new__(cls)
        obj.cur_vehicle_state = Vehicle.fromRosMsg(msg.cur_vehicle_state)
        obj.current_lane = CurrentLane.fromRosMsg(msg.current_lane)
        obj.next_intersection = [
            PerpendicularLane.fromRosMsg(pline) for pline in msg.next_intersection
        ]
        obj.adjacent_lanes = [
            ParallelLane.fromRosMsg(pline) for pline in msg.adjacent_lanes
        ]
        obj.speed_limit = msg.speed_limit
        obj.reward_info = RewardInfo.fromRosMsg(msg.reward_info)
        obj.global_path = GlobalPath.fromRosMsg(msg.global_path)
        return obj

    def toRosMsg(self):
        msg = EnvDescMsg()
        msg.cur_vehicle_state = self.cur_vehicle_state.toRosMsg()
        msg.current_lane = self.current_lane.toRosMsg()
        msg.next_intersection = [line.toRosMsg() for line in self.next_intersection]
        msg.adjacent_lanes = [line.toRosMsg() for line in self.adjacent_lanes]
        msg.speed_limit = self.speed_limit
        msg.reward_info = self.reward_info.toRosMsg()
        msg.global_path = self.global_path.toRosMsg()
        return msg


TEST_NODE_NAME = "custom_listener"
TEST_TOPIC_NAME = "custom_chatter"

# Test Code for Deserialize and Infor Printing
def callback(data):
    obj = EnvDesc.fromRosMsg(data)
    pl = obj.next_intersection[0]
    print(pl.__dict__)
    for p in pl.crossing_pedestrain:
        print(p.location_global.x)
        print(p.location_global.y)
        print(p.location_global.theta)
    print(pl.intersecting_distance)
    print(pl.origin_global_pose.x)
    print(pl.origin_global_pose.y)
    print(pl.origin_global_pose.theta)


def listener():
    rospy.init_node(TEST_NODE_NAME, anonymous=True)
    rospy.Subscriber(TEST_TOPIC_NAME, EnvDescMsg, callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()


if __name__ == "__main__":
    listener()
