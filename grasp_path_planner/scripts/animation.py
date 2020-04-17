from settings import *
# For Visualisation
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import json
import numpy as np
obj=None
fig = plt.figure()
ax = fig.add_subplot(1,1,1)

def animate_callback(i):
    print("ENTERED CALLBACK")
    action_probs = np.load("action_probs.npz")["action_probs"]
    ax.clear()
    if CURRENT_SCENARIO == Scenario.LANE_CHANGE:
        ax.bar(x=[1,2,3,4], height = action_probs, tick_label=["constant","accelerate","decelerate","lane_change"])
    else:
        ax.bar(x=[1,2,3], height = action_probs, tick_label=["constant","accelerate","decelerate"])

if __name__ == "__main__":
    obj=animation.FuncAnimation(fig, animate_callback, interval=1000)
    plt.show()