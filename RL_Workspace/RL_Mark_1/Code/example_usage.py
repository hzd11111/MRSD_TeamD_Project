import argparse
import logging
import numpy as np
from random import seed
from timeit import default_timer as timer
import time
import pygame

import gym
import gym_racer
from gym_racer.envs.utils import getMyLogger


def parse_arguments():
    """Setup CLI interface
    """
    parser = argparse.ArgumentParser(description="Test the racer env")

    parser.add_argument("-fps", "--fps", type=int, default=1, help="frame per second")
    parser.add_argument(
        "-s", "--rand_seed", type=int, default=-1, help="random seed to use"
    )
    parser.add_argument(
        "-nf",
        "--num_frames",
        type=int,
        default=-1,
        help="how many frames to run, -1 is unlimited",
    )
    parser.add_argument(
        "-i", "--interactive", action="store_true", help="start interactive env"
    )

    # last line to parse the args
    args = parser.parse_args()
    return args


def setup_logger(logLevel="DEBUG"):
    """Setup logger that outputs to console for the module
    """
    logroot = logging.getLogger("c")
    logroot.propagate = False
    logroot.setLevel(logLevel)

    module_console_handler = logging.StreamHandler()

    #  log_format_module = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    #  log_format_module = "%(name)s - %(levelname)s: %(message)s"
    #  log_format_module = '%(levelname)s: %(message)s'
    #  log_format_module = "%(name)s: %(message)s"
    log_format_module = "%(message)s"

    formatter = logging.Formatter(log_format_module)
    module_console_handler.setFormatter(formatter)

    logroot.addHandler(module_console_handler)

    logging.addLevelName(5, "TRACE")
    # use it like this
    # logroot.log(5, 'Exceedingly verbose debug')

    # example log line
    logg = logging.getLogger(f"c.{__name__}.setup_logger")
    logg.setLevel("INFO")
    logg.debug(f"Done setting up logger")


def setup_env():
    setup_logger()

    args = parse_arguments()

    # setup seed value
    if args.rand_seed == -1:
        myseed = 1
        myseed = int(timer() * 1e9 % 2 ** 32)
    else:
        myseed = args.rand_seed
    seed(myseed)
    np.random.seed(myseed)

    # build command string to repeat this run
    # NOTE this does not work for flags, but whatever
    recap = f"python3 test_env.py"
    for a, v in args._get_kwargs():
        if a == "rand_seed":
            recap += f" --rand_seed {myseed}"
        else:
            recap += f" --{a} {v}"

    logmain = logging.getLogger(f"c.{__name__}.setup_env")
    logmain.info(recap)

    return args


def test_interactive_env(args):
    """
    """
    logg = getMyLogger(f"c.{__name__}.test_interactive_env", "DEBUG")
    logg.info(f"Start test_interactive_env")

    fps = args.fps
    num_frames = args.num_frames

    racer_env = gym.make("racer-v1", render_mode="human")

    # clock for interactive play
    clock = pygame.time.Clock()

    # Main Loop
    going = True
    i = 0
    while going:
        logg.info(f"----------    ----------    New frame    ----------    ----------")

        start_frame = timer()

        # Handle Input Events
        # https://stackoverflow.com/a/22099654
        for event in pygame.event.get():
            #  logg.debug(f"Handling event {event}")
            if event.type == pygame.QUIT:
                going = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    going = False
            #  logg.debug(f"Done handling")

        keys = pygame.key.get_pressed()
        if keys[pygame.K_d]:  # right
            action = [0, 2]
        elif keys[pygame.K_a]:  # left
            action = [0, 1]
        elif keys[pygame.K_w]:  # up
            action = [1, 0]
        elif keys[pygame.K_x]:  # down
            action = [2, 0]
        elif keys[pygame.K_q]:  # upleft
            action = [1, 2]
        elif keys[pygame.K_e]:  # upright
            action = [1, 2]
        elif keys[pygame.K_z]:  # downleft
            action = [2, 1]
        elif keys[pygame.K_c]:  # downright
            action = [2, 2]
        else:  # nop
            action = [0, 0]

        logg.debug(f"Do the action {action}")

        mid_frame = timer()

        # perform the action
        obs, reward, done, info = racer_env.step(action)

        step_frame = timer()

        # draw the new state
        racer_env.render(mode="human", reward=reward)

        end_frame = timer()

        logg.debug(f"Time for input  {mid_frame-start_frame:.6f} s")
        logg.debug(f"Time for step   {step_frame-mid_frame:.6f} s")
        logg.debug(f"Time for render {end_frame-step_frame:.6f} s")
        logg.debug(f"Time for frame  {end_frame-start_frame:.6f} s")

        # wait a bit to limit fps
        clock.tick(fps)

        if num_frames > 0:
            i += 1
            if i == num_frames:
                going = False


def test_automatic_env(args):
    """
    """
    logg = getMyLogger(f"c.{__name__}.test_automatic_env", "DEBUG")
    # logg = getMyLogger(f"c.{__name__}.test_automatic_env", "INFO")
    logg.info(f"Start test_automatic_env")

    fps = args.fps
    num_frames = args.num_frames
    clock = pygame.time.Clock()

    mode = "human"
    #  mode = "console"
    #  sat = "diamond"
    sat = "lidar"

    sensor_array_params = {}
    sensor_array_params["ray_num"] = 7
    sensor_array_params["ray_step"] = 10
    sensor_array_params["ray_sensors_per_ray"] = 13
    sensor_array_params["ray_max_angle"] = 130
    sensor_array_params["viewfield_size"] = 30
    sensor_array_params["viewfield_step"] = 8

    racer_env = gym.make(
        "racer-v1",
        sensor_array_type=sat,
        render_mode=mode,
        sensor_array_params=sensor_array_params,
    )

    logg.info(f"Action Space {racer_env.action_space}")
    logg.info(f"State Space {racer_env.observation_space}")

    going = True
    i = 0
    tot_frame_times = 0
    tot_step_times = 0
    tot_render_times = 0
    while going:
        start = timer()
        pygame.event.get()
        end = timer()
        logg.debug(f"----------    ----------    New frame    ----------    ----------")

        t01 = timer()

        action = racer_env.action_space.sample()

        t02 = timer()

        obs, reward, done, info = racer_env.step(action)

        t03 = timer()
        # pygame.display.update()
        racer_env.render(mode=mode, reward=reward)

        t04 = timer()

        logg.debug(f"Do the action {action}")
        logg.debug(f"obs shape {obs.shape}")

        logg.debug(f"Time for sample {t02-t01:.6f} s")

        step_time = t03 - t02
        logg.debug(f"Time for step   {step_time:.6f} s")
        tot_step_times += step_time

        render_time = t04 - t03
        logg.debug(f"Time for render {render_time:.6f} s")
        tot_render_times += render_time

        frame_time = t04 - t01
        logg.debug(f"Time for frame  {frame_time:.6f} s")
        tot_frame_times += frame_time

        #  clock.tick(fps)

        recap = ""
        recap += f"Car state: x {info['car_pos_x']}"
        recap += f" y {info['car_pos_y']}"
        recap += f" dir {info['car_dir']}"
        recap += f" speed {info['car_speed']}"
        recap += f"\t\treward {reward}"
        logg.debug(recap)

        going = not done
        # print(going)
        i += 1
        if num_frames > 0:
            if i == num_frames:
                going = False

    mean_step_time = tot_step_times / i
    logg.info(f"Average time for step   {mean_step_time:.6f} s")

    mean_render_time = tot_render_times / i
    logg.info(f"Average time for render {mean_render_time:.6f} s")

    mean_frame_time = tot_frame_times / i
    logg.info(f"Average time for frame  {mean_frame_time:.6f} s")


if __name__ == "__main__":
    args = setup_env()
    if args.interactive:
        test_interactive_env(args)
    else:
        test_automatic_env(args)

