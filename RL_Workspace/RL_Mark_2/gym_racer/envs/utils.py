import logging
import numpy as np

from math import cos
from math import sin
from math import radians


def getMyLogger(logger_name, log_level="DEBUG"):
    """returns the logger with the requested level
    """
    logg = logging.getLogger(logger_name)
    logg.setLevel(log_level)
    return logg


def compute_rot_matrix(theta):
    """compute the rotation matrix for angle theta in degrees
    """
    #  logg = getMyLogger(f"c.{__name__}.compute_rot_matrix", "INFO")

    theta = radians(theta)
    ct = cos(theta)
    st = sin(theta)
    rot_mat = np.array(((ct, -st), (st, ct)))
    #  logg.debug(f"rot_mat = {rot_mat}")

    return rot_mat
