#!/home/bitdrones/syscon_2025/syscon/bin/python3

import rclpy
from rclpy.node import Node
import numpy as np
import time
from geometry_msgs.msg import PoseStamped, TwistStamped, Twist
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from rclpy.duration import Duration
from motion_capture_tracking_interfaces.msg import NamedPoseArray
from visualization_msgs.msg import Marker
from std_msgs.msg import Float32MultiArray, Bool
from std_srvs.srv import Empty
from scipy.spatial.transform import Rotation as R
from scipy.stats import multivariate_normal
from target_tracking.core import ExclusionZone
from target_tracking.utils import *
import os
os.environ["PYTHONPATH"] = "/home/bitdrones/syscon_2025/syscon/lib/python3.10/site-packages"

experiment = {
    "dt": 0.1,
    "env_size": 100,
    "win_radius": 5,
    "evaluation_episodes": 10,
    "type": "dynamic",
    "subtype": "equal",
    "H": 10,  # prediction horizon
    "weightsEqual": [1/7] * 7,
    "weightsA": [0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1],
    "weightsB": [0.1, 0.1, 0.1, 0.25, 0.25, 0.1, 0.1],
    "weightsC": [0.1, 0.1, 0.1, 0.1, 0.1, 0.25, 0.25],
}

commands = Twist()
commandsS = TwistStamped()
internal_values = TwistStamped()


class CommandsNode(Node):
    def __init__(self):
        super().__init__('commands_node')
        self.declare_parameter('robot', 'C20')
        self.declare_parameter('n_agents', 3)
        self.declare_parameter('hover_height', 0.5)
        self.robot = self.get_parameter('robot').value
        self.n_obstacles = int(self.get_parameter('n_agents').value) - 1
        self.hover_height = self.get_parameter('hover_height').value

        self.obs_size = 2*self.n_obstacles + 3
        self.obs = np.zeros((1, self.obs_size))
        self.drone_pose = np.zeros(3)
        self.target = np.zeros(3)
        self.agents = np.zeros((self.n_obstacles, 3))
        self.vel = np.array([1, 0])
        self.heading = np.arctan2(self.vel[1], self.vel[0]) / np.pi
        self.has_poses = False
        self.has_taken_off = False
        self.has_landed = False
        self.land_flag = False
        self.start = False
        self.commands = Twist()

        self.commands_pub = self.create_publisher(Twist, '/'+ self.robot +'/cmd_vel_slow', 1)
        self.commands_pubS = self.create_publisher(TwistStamped, '/'+ self.robot +'/cmd_vel_stamped', 1)
        # self.internal_values_pub = self.create_publisher(TwistStamped, '/internal_values', 1)
        qos_profile = QoSProfile(reliability =QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            deadline=Duration(seconds=0, nanoseconds=0))
        self.create_subscription(
            NamedPoseArray, "/poses",
            self._pose_callback, qos_profile
        )
        self.subscription = self.create_subscription(
            Bool,
            '/landing',
            self._landing_callback,
            10)
        self.subscription = self.create_subscription(
            Bool,
            '/encircle',
            self._start_callback,
            10)
        # while not self.has_poses:
        #     rclpy.spin_once(self, timeout_sec=0.1)

        self.OBSTs = []
        radius = 0.5
        for i in range(self.n_obstacles):
            ez_name = f"OBST_{i}"
            self.OBSTs.append(ExclusionZone(radius, 5, ez_name))

            self.OBSTs[i].pos[0] = self.agents[i,0]
            self.OBSTs[i].pos[1] = self.agents[i,1]


        dir_models = "/home/bitdrones/syscon_2025/Models/models_2_4_5/two_obst/2024-10-30_16-50-13"
        self.tactics = Tactic(None, os.path.join(dir_models, "two_obst"), 1)

        self.get_logger().info(f"tactic: {self.tactics}")
        # self.selected_tactic = np.zeros(100000) - 1

        self.t = 0
        self.timer_period = 0.01
        self.t_init = time.time()
        self.get_logger().info("Starting")
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

    def _pose_callback(self, msg):
        self.has_poses = True
        i = 0
        for pose in msg.poses:
            if pose.name == self.robot:
                heading = R.from_quat([pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z, pose.pose.orientation.w]).as_euler('zyx')[0]
                self.drone_pose = np.array([pose.pose.position.x, pose.pose.position.y, heading])
            elif pose.name == "QCar":
                heading = R.from_quat([pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z, pose.pose.orientation.w]).as_euler('zyx')[0]
                self.target = np.array([pose.pose.position.x, pose.pose.position.y, heading])
            else:
                heading = R.from_quat([pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z, pose.pose.orientation.w]).as_euler('zyx')[0]
                self.agents[i] = np.array([pose.pose.position.x, pose.pose.position.y, heading])
                i += 1
                

    def timer_callback(self):
        if self.land_flag:
            commands = Twist()
            commands.linear.z = -0.5
            self.commands_pub.publish(commands)
            if (time.time() - self.t_init_land) > 3:
                self.has_landed = True
                self.reboot()
        else:
            if not self.has_taken_off:
                commands = Twist()
                if self.drone_pose[2] < self.hover_height:
                    commands.linear.z = 0.5
                    self.commands_pub.publish(commands)
                if ((time.time() - self.t_init) > 3) and self.start:
                    self.has_taken_off = True
            else:
                self.obs = self.generate_observation(self.OBSTs, self.drone_pose, self.target)
                self.calc_vel()
        
       

    def reboot(self):
        req = Empty.Request()
        self.reboot_client.call_async(req)
        time.sleep(2.0)
    def _landing_callback(self, msg):
        self.land_flag = msg.data
        self.t_init_land = time.time()
    def _start_callback(self, msg):
        self.start = msg.data

    def generate_observation(self, OBSTs, pursuer, evader):
        stateE = evader
        maxD = 2 * 5 * np.sqrt(2)

        observation = []
        stateP = pursuer

        pos_pur_eva = stateE[0:2] - stateP[0:2]
        dist_pur_eva = np.linalg.norm(pos_pur_eva)
        angle = np.arctan2(pos_pur_eva[1], pos_pur_eva[0])
        dist_pur_eva /= maxD
        angle /= np.pi
        pTheta = stateP[2] / np.pi

        dPEzs = []
        angPEzs = []

        for ez in OBSTs:
            vecPEzi = ez.pos[0:2] - stateP[0:2]
            dwez = np.clip((np.linalg.norm(vecPEzi) - ez.radius) / maxD, -1, 1)
            dPEzs.append(dwez)
            angPEzs.append(np.arctan2(vecPEzi[1], vecPEzi[0]) / np.pi)

        dist_pur_eva = np.clip(dist_pur_eva, 0, 1)
        observation.append([dist_pur_eva, angle, pTheta, dPEzs, angPEzs])

        flattened_observation = flatten(observation)
        observation_array = np.array(flattened_observation)

        return observation_array

    def calc_vel(self):
        vel = 0.1

        angular_velocity = self.tactics.compute_Action(self.obs).squeeze()
        self.get_logger().info(f"angular_velocity: {angular_velocity}")
        u = np.arctan2(self.vel[1], self.vel[0]) + 0.1 * angular_velocity * np.pi
        u = np.arctan2(np.sin(u), np.cos(u))
        vx = vel * np.cos(u)
        vy = vel * np.sin(u)
        self.commands.linear.x = vx
        self.commands.linear.y = vy
        self.commands_pub.publish(commands)

        commandsS.twist.linear.x = vx
        commandsS.twist.linear.y = vy
        commandsS.header.stamp = self.get_clock().now().to_msg()
        commandsS.header.frame_id = self.robot
        self.commands_pubS.publish(commandsS)


        self.vel = np.array([vx, vy])


def main(args=None):
    rclpy.init()
    commander = CommandsNode()
    rclpy.spin(commander)
    commander.destroy_node()
    rclpy.shutdown()



if __name__ == '__main__':
    main()
