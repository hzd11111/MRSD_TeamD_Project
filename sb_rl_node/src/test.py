import rosbag
import torch
import dqn_manager
# from msg.VehicleState import VehicleState

messages = rosbag.Bag("msg.bag")
env_msgs = [msg for _,msg,_ in messages.read_messages(['/environment_state'])]
data = env_msgs[0]
env_state = []

def append_vehicle_state(env_state, vehicle_state):
    env_state.append(vehicle_state.vehicle_location.x)
    env_state.append(vehicle_state.vehicle_location.y)
    env_state.append(vehicle_state.vehicle_location.theta)
    env_state.append(vehicle_state.vehicle_speed)
    return
    
# items needed
# current vehicle velocity
env_state.append(data.cur_vehicle_state.vehicle_speed)
# rear vehicle state
    # position
    # velocity
append_vehicle_state(env_state, data.back_vehicle_state)
# adjacent vehicle state
    # position
    # velocity
for veh_state in data.adjacent_lane_vehicles:
    append_vehicle_state(env_state, veh_state)

def make_state_vector(data):
    '''
    create a state vector from the message recieved
    '''
    env_state = []
    def append_vehicle_state(env_state, vehicle_state):
        env_state.append(vehicle_state.vehicle_location.x)
        env_state.append(vehicle_state.vehicle_location.y)
        env_state.append(vehicle_state.vehicle_location.theta)
        env_state.append(vehicle_state.vehicle_speed)
        return
    # items needed
    # current vehicle velocity
    env_state.append(data.cur_vehicle_state.vehicle_speed)
    # rear vehicle state
        # position
        # velocity
    append_vehicle_state(env_state, data.back_vehicle_state)
    # adjacent vehicle state
        # position
        # velocity
    i = 0
    for i, veh_state in enumerate(data.adjacent_lane_vehicles):
        if i <= 5:
            append_vehicle_state(env_state, veh_state)
        else:
            break
    
    dummy = data.cur_vehicle_state
    dummy.vehicle_location.x = 10000
    dummy.vehicle_location.y = 10000
    dummy.vehicle_location.theta = 10000
    dummy.vehicle_speed = 0
    while i<=5:
        append_vehicle_state(env_state, dummy)
        i+=1
    return env_state

state = make_state_vector(env_msgs[-50])
state_tensor = torch.Tensor(state).to("cpu")
settings = {
	"BATCH_SIZE" : 64,
	"GAMMA" : 0.99,
	"EPS_START" : 0.9,
	"EPS_END" : 0.05,
	"EPS_DECAY" : 200,
	"TARGET_UPDATE" : 10,
	"INPUT_HEIGHT" : 0,
	"INPUT_WIDTH" : 29,
	"CAPACITY" : 10000,
	"N_ACTIONS": 4,
	"DEVICE" : "cpu"
}
manager = dqn_manager.DQNManager(*settings.values())
manager.initialize()
arg_val = manager.target_net(state_tensor).view(-1,4).max(1)[1].view(1,1)
print(arg_val.item())



