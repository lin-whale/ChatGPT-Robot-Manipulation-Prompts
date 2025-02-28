import sys
from app_msgs.srv import StorageItem, RightArmDetect
import app_msgs.msg
import app_msgs.srv
import app_msgs
from dual_arm_interfaces.action import ArmTask
from dual_arm_interfaces.msg import PerceptionInformation, PerceptionTarget
from geometry_msgs.msg import Pose, Point
import rclpy
import rclpy.logging
from rclpy.node import Node
from rclpy.action import ActionClient, ActionServer
import time
from rclpy.executors import MultiThreadedExecutor

class VirtualExecutor(Node):
    def __init__(self):
        super().__init__('virtual_executor')
        self.chassis_exec_service = self.create_service(StorageItem, "srv_search_in_map", callback=self.chassis_exec)
        self.head_detect_service = self.create_service(app_msgs.srv.SearchSomething, "srv_head_detect_name", callback=self.head_detect)
        self.head_detect_pub = self.create_publisher(app_msgs.msg.SearchSomething, "yolo_world_search_pub", 10)
        self.arm_detect_service = self.create_service(RightArmDetect, "srv_arm_detect", callback=self.arm_detect)
        self.arm_detect_sub = self.create_subscription(PerceptionInformation, "topic_arm_detect", self.arm_detect_sub_callback, 10)
        self.arm_exec_action = ActionServer(self, ArmTask, "action_arm_grasp", self.arm_execute_callback)
        self.arm_detect_result = None

    def chassis_exec(self, request, response):
        self.get_logger().info('--------------------> ' + "Chassis Got req:")
        self.get_logger().info(" ".join(request.item_labels))
        if request.type == 0:
            time.sleep(4)
            self.get_logger().info("All room search success!")
        elif request.type == -2:
            time.sleep(5)
            self.get_logger().info("Spefied object search and move success!")
        else:
            raise RuntimeError("Unsupported task type {}".format(request.type))
        response.result = True
        return response
    
    def head_detect(self, request, response):
        self.get_logger().info('-------------------->' + " Head detect Got req:")
        self.get_logger().info(request.searchsomething_labels)
        time.sleep(5)
        timer_period = 0.5
        self.timer = self.create_timer(timer_period, callback=self.head_detect_callback)
        response.result = True
        return response
    
    def head_detect_callback(self):
        msg = app_msgs.msg.SearchSomething()
        self.head_detect_pub.publish(msg)
    
    def arm_detect(self, request, response):
        self.get_logger().info('--------------------> ' + "Arm detect Got req:")
        self.get_logger().info(request.label)
        response.target_poses = [Pose()]
        response.result_labels = [request.label]
        return response
    
    def arm_detect_sub_callback(self, msg):
        self.arm_detect_result = msg.right_hand_camera
        # self.get_logger().info('-----> ' + 'Set arm_detect_result')

    def arm_execute_callback(self, goal_handle):
        self.get_logger().info('--------------------> ' + "Executing goal...")
        for _ in range(10):
            # print(self.arm_detect_result)
            self.get_logger().info()
            time.sleep(1)
        goal_handle.succeed()
        result = ArmTask.Result()
        return result

def main():
    rclpy.init()
    node = VirtualExecutor()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        executor.shutdown()
        node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()