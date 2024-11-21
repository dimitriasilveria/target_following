from std_msgs.msg import String
from geometry_msgs.msg import Twist
from rclpy.qos import QoSProfile
from rclpy.node import Node
import rclpy


class FastCMDVel(Node):
    def __init__(self):
        super().__init__('FastCMDVel')
        self.declare_parameter('robot', 'C20')
        self.robot = self.get_parameter('robot').value
        self.create_subscription(Twist, '/'+ self.robot +'/cmd_vel_fast', self.callbackSlowMessage, 10)
        self.vel_pub = self.create_publisher(Twist, '/'+ self.robot +'/cmd_velocity', 10)
        self.cmd = Twist()
        timer_period = 0.01
        self.create_timer(timer_period,self.timer_callback)
    
    def callbackSlowMessage(self, data):
        self.cmd = data

    def timer_callback(self):
        self.vel_pub.publish(self.cmd)

def main(args=None):
    rclpy.init(args=args)
    fast_cmd_vel = FastCMDVel()
    rclpy.spin(fast_cmd_vel)
    fast_cmd_vel.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()