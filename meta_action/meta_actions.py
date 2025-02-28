import sys
from app_msgs.srv import StorageItem, RightArmDetect
import app_msgs
import geometry_msgs
import app_msgs.msg
import app_msgs.srv
from dual_arm_interfaces.action import ArmTask
from dual_arm_interfaces.msg import PerceptionInformation, PerceptionTarget
from geometry_msgs.msg import Pose, Point
import geometry_msgs.msg
from std_msgs.msg import String, Bool
import rclpy
import rclpy.logging
from rclpy.node import Node
from rclpy.action import ActionClient
import time
from .utils import Point, Pose2D, camera_to_world_pose

class PlanExecutor(Node):

    def __init__(self, task_sequence: list, obj_name: str):
        super().__init__('plan_controller')
        self.task_sequence = task_sequence
        self.obj_name = obj_name

        self.search_in_map_cli = self.create_client(StorageItem, '/storage_item_service')
        self.arm_action_cli = ActionClient(self, ArmTask, 'action_arm_grasp')
        self.arm_detect_cli = self.create_client(RightArmDetect, 'srv_arm_detect')
        self.detect_publisher = self.create_publisher(PerceptionInformation, 'topic_arm_detect', 10)

        # while not self.arm_action_cli.wait_for_server(timeout_sec=1.0):
        #     self.get_logger().info('Arm_grasp action server not available, waiting again...')

        while not self.search_in_map_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('/storage_item_service not available, waiting again...')

        # while not self.arm_detect_cli.wait_for_service(timeout_sec=1.0):
        #     self.get_logger().info('Arm detect service not available, waiting again...')

    def execute(self):
        try:
            for task in self.task_sequence:
                if task == "search_roughly_and_approach_object()":
                    self.action_reuslt = self.send_search_in_map(obj_name=self.obj_name, task_type=0) # "all room"
                if task == "move_chassis_based_on_object()":
                    self.action_reuslt = self.send_search_in_map(obj_name=self.obj_name, task_type=-2) # "specified object"
                if task == "detect_precisely()":
                    self.action_reuslt = self.arm_detect(label=self.obj_name)
                if task == "grasp_object()":
                    self.action_reuslt = self.grasp(241)    # take photos 240; grasp 241.
                else:
                    raise RuntimeError("Wrong task name {}!".format(task))
        except Exception as e:
            return (task, e)
        print("All task executed successfully!")
    
    def send_search_in_map(self, obj_name, task_type):
        """send StorageItem srv to chassis control, including "all room" and "a specified object" task."""
        search_req = StorageItem.Request()
        search_req.item_labels = [obj_name]
        search_req.task_state = 1
        search_req.type = task_type
        search_req.item_pose = geometry_msgs.msg.Point(x=0, y=0, z=0)  # pseudo item_pose
        if task_type == 0:
            # Start head camera detection service.
            self.head_detect_pub = self.create_publisher(Bool, "/task_switch_topic", 10)
            head_detect_bool_msg = Bool()
            head_detect_bool_msg.data = True
            self.head_detect_pub.publish(head_detect_bool_msg)
        elif task_type == -2:
            # Complete search_req.item_pose
            self.head_detect_result = None
            self.chassis_pose = None
            self.head_detect_sub = self.create_subscription(app_msgs.msg.SearchSomething, "/yolo_world_search_pub", self.head_detect_callback, 10)
            self.chassis_pose_sub = self.create_subscription(geometry_msgs.msg.PoseStamped, "/tracked_pose", self.tracked_pose_callback, 10)
            while self.head_detect_result is None or self.chassis_pose is None:
                rclpy.spin_once(self)
            
            # construct chassis_pose and point data
            camera_item_point = Point(self.head_detect_result.x, self.head_detect_result.y, self.head_detect_result.z)
            chass_orientation = self.chassis_pose.pose.orientation
            chassis_pose = Pose2D(self.chassis_pose.pose.position.x, self.chassis_pose.pose.position.y) \
                                  .from_quat(chass_orientation.x, chass_orientation.y, chass_orientation.z, chass_orientation.w)
            world_item_pose = camera_to_world_pose(chassis_pose, camera_item_point)
            search_req.item_pose = geometry_msgs.msg.Point(x=world_item_pose.x, y=world_item_pose.y, z=world_item_pose.z)
        else:
            raise RuntimeError("Unsupported task_type: {}".format(task_type))

        return self.send_ros2_service_request(self.search_in_map_cli, search_req)
    
    def tracked_pose_callback(self, msg):
        self.chassis_pose = msg
        self.destroy_subscription(self.chassis_pose_sub)

    def head_detect_callback(self, msg):
        if self.obj_name in msg.searchsomething_labels:
            index = msg.searchsomething_labels.index(self.obj_name)
            self.head_detect_result = msg.something_pose[index]     # geometry_msgs/Point msg, has attribute float64 x, y, z
            self.destroy_subscription(self.head_detect_sub)
    
    def grasp(self, task_id: int):
        goal_msg = ArmTask.Goal()
        goal_msg.task_id = task_id
        result = self.send_ros2_action_request(self.arm_action_cli, goal_msg)
        return result
    
    def arm_detect(self, label: str):
        detect_freq = 0.2
        self.timer = self.create_timer(detect_freq, lambda: self.send_request_detect(label=label))

    def send_request_detect(self, label: str):
        req = RightArmDetect.Request()
        req.label = label
        self.future = self.arm_detect_cli.call_async(req)
        self.future.add_done_callback(self.detect_callback)

    def detect_callback(self, future):
        try:
            response = future.result()
            perception_target = PerceptionTarget()
            perception_target.target_pose = response.target_poses[0]
            perception_target.label = String()
            perception_target.label.data = response.result_labels[0]
            perception_info = PerceptionInformation()
            perception_info.right_hand_camera = [perception_target]
            self.detect_publisher.publish(perception_info)
        except Exception as e:
            self.get_logger().error('Detect service call failed {}'.format(e))
    
    def send_ros2_service_request(self, cli, req):
        """Send ros2 service requests synchronously and get response.
        Args:
            cli:    client to send service request.
            req:    request messages.
        Returns:
            result received from server.
        """
        future = cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        return future.result()
    
    def send_ros2_action_request(self, cli: ActionClient, req):
        """Send ros2 action requests and get result synchronously.
        Args:
            cli:    action client.
            req:    action request.
        Returns:
            result received from action server.
        """
        cli.wait_for_server()
        self._send_goal_future = cli.send_goal_async(req)
        rclpy.spin_until_future_complete(self, self._send_goal_future)
        goal_handle = self._send_goal_future.result()
        self._get_result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, self._get_result_future)
        result = self._get_result_future.result().result
        self.get_logger().info('Result: {0}'.format(result.result))
        return result.success


def search_in_map(object: str):
    """Find an object in global map, return the location of the found object.
    Args:
        object(str): The name of the object to search for.
    
    Returns:
        list: The location of the object as a list, or None if not found.

    Raises:
        TypeError: If the input 'object' is not a string.
    """
    if not isinstance(object, str):
        raise TypeError('Input object is not a string!')
    print("search_in_map is executing...")
    time.sleep(5)
    return [1, 2, 3]


def move_chassis(loc: list):
    """Move chassis to approach an object.
    Args:
        loc(list): The location of the object.

    Returns:
        bool: indecate the task is successful or not.
    """
    if not isinstance(loc, list):
        raise TypeError('Loc must be list!')
    print("move_object is executing...")
    time.sleep(3)
    return True


def detect_for_grasp(object: str):
    """Detect object for grasp, return precise boundingbox of the object.
    Args:
        object(str): The name of the object to detect for grasp.

    Returns:
        list: Boundingbox of the detected object.
    """
    print("detect_for_grasp executing...")
    time.sleep(4)
    return [1.0, 1.0, 2.0, 3.0]


def grasp_object(object: str, bbox:list):
    """Grasp object.
    Args:
        object(str): The name of the object to grasp.
        bbox(list): The boundingbox of the object.

    Returns:
        bool: result of the grasp.
    """
    print("grasp_object is executing...")
    time.sleep(10)
    return True
