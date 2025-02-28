import numpy as np
import math
from scipy.spatial.transform import Rotation as R

class Point:
    def __init__(self, x, y, z=0.0):
        self.x = x
        self.y = y
        self.z = z

class Pose2D:
    def __init__(self, x=0, y=0, theta=0):
        self.x = x
        self.y = y
        self.theta = theta
    
    def from_quat(self, quaternion):
        r = R.from_quat(quaternion)
        euler_angles_rad = r.as_euler('xyz')
        self.theta = euler_angles_rad[-1]


def camera_to_world_pose(robot_pose, camera_item_pose):
    """
    将相机坐标系下的物品位置转换为世界坐标系下的位置。

    Args:
        robot_pose (Pose2D): 机器人位姿。
        camera_item_pose (Point): 相机坐标系下的物品位置。

    Returns:
        Point: 世界坐标系下的物品位置。
    """
    camera_to_robot = np.array([1.5707963267949, -3.1415926535898, 1.0112488854614])

    # 旋转矩阵
    rotate_z = np.array([[math.cos(camera_to_robot[0]), -math.sin(camera_to_robot[0]), 0],
                         [math.sin(camera_to_robot[0]), math.cos(camera_to_robot[0]), 0],
                         [0, 0, 1]])

    rotate_y = np.array([[math.cos(camera_to_robot[1]), 0, math.sin(camera_to_robot[1])],
                         [0, 1, 0],
                         [-math.sin(camera_to_robot[1]), 0, math.cos(camera_to_robot[1])]])

    rotate_x = np.array([[1, 0, 0],
                         [0, math.cos(camera_to_robot[2]), -math.sin(camera_to_robot[2])],
                         [0, math.sin(camera_to_robot[2]), math.cos(camera_to_robot[2])]])

    # 组合旋转矩阵
    R = np.dot(np.dot(rotate_z, rotate_y), rotate_x)

    point_camera = np.array([camera_item_pose.x, camera_item_pose.y, camera_item_pose.z])
    point_robot = np.dot(R, point_camera)

    point_robot[0] += 0.10652561
    point_robot[1] += 0.007054035
    point_robot[2] += 1.105

    print(f"point_robot x: {point_robot[0]}")
    print(f"point_robot y: {point_robot[1]}")

    item_pose = Point(0, 0, 0)
    item_pose.x = point_robot[0] * math.cos(robot_pose.theta) - point_robot[1] * math.sin(robot_pose.theta) + robot_pose.x
    item_pose.y = point_robot[0] * math.sin(robot_pose.theta) + point_robot[1] * math.cos(robot_pose.theta) + robot_pose.y
    item_pose.z = point_robot[2]

    return item_pose
