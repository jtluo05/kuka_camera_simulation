"""Tracking controller: drives lbr_A1 (pan) and lbr_A6 (tilt) to keep the
detected face centered in the camera image, with configurable laser offset bias.

x_bias: positive = target moves RIGHT in image (nose left of laser = laser right of nose)
y_bias: positive = target moves DOWN in image (nose above laser = laser below nose)

For laser 1 inch below and right of nose:
  set x_bias negative (move target left) and y_bias negative (move target up)
"""
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

JOINT_NAMES = ['lbr_A1', 'lbr_A2', 'lbr_A3', 'lbr_A4',
               'lbr_A5', 'lbr_A6', 'lbr_A7']
LIMITS = {'lbr_A1': 2.9, 'lbr_A6': 2.0}

class FaceTrackerController(Node):
    def __init__(self):
        super().__init__('face_tracker_controller')
        self.declare_parameter('pan_gain', -0.8)
        self.declare_parameter('tilt_gain', -0.8)
        self.declare_parameter('deadband', 0.02)
        self.declare_parameter('max_step', 0.04)
        self.declare_parameter('rate_hz', 10.0)
        self.declare_parameter('error_timeout', 0.5)
        # Laser offset bias in normalized image coordinates (0.0-1.0 range)
        # x_bias: positive shifts aim RIGHT, negative shifts aim LEFT
        # y_bias: positive shifts aim DOWN, negative shifts aim UP
        # Start at 0.0 and tune at runtime with --ros-args -p x_bias:=VALUE
        self.declare_parameter('x_bias', 0.0)
        self.declare_parameter('y_bias', 0.0)

        self.latest_error = None
        self.latest_error_time = None
        self.joint_pos = {}

        self.create_subscription(Point, '/face_tracking/error',
                                 self.error_cb, 10)
        self.create_subscription(JointState, '/lbr/joint_states',
                                 self.joints_cb, 10)
        self.traj_pub = self.create_publisher(
            JointTrajectory,
            '/lbr/joint_trajectory_controller/joint_trajectory', 10)

        rate = self.get_parameter('rate_hz').value
        self.timer = self.create_timer(1.0 / rate, self.control_tick)
        self.get_logger().info('Face tracker controller running.')

    def error_cb(self, msg: Point):
        self.latest_error = msg
        self.latest_error_time = self.get_clock().now()

    def joints_cb(self, msg: JointState):
        for name, pos in zip(msg.name, msg.position):
            self.joint_pos[name] = pos

    def control_tick(self):
        if len(self.joint_pos) < len(JOINT_NAMES):
            return
        if self.latest_error is None or self.latest_error.z < 0.5:
            return
        age = (self.get_clock().now() - self.latest_error_time).nanoseconds / 1e9
        if age > self.get_parameter('error_timeout').value:
            return

        err = self.latest_error
        x_bias = self.get_parameter('x_bias').value
        y_bias = self.get_parameter('y_bias').value

        # Apply bias: shift the target point in the image
        # Laser is 1 inch below and right of nose, so we want nose
        # to appear ABOVE and LEFT of center -> negative bias on both
        adjusted_x = err.x + x_bias
        adjusted_y = err.y + y_bias

        deadband = self.get_parameter('deadband').value
        max_step = self.get_parameter('max_step').value

        d_pan = self.get_parameter('pan_gain').value * adjusted_x \
            if abs(adjusted_x) > deadband else 0.0
        d_tilt = self.get_parameter('tilt_gain').value * adjusted_y \
            if abs(adjusted_y) > deadband else 0.0

        d_pan  = max(-max_step, min(max_step, d_pan))
        d_tilt = max(-max_step, min(max_step, d_tilt))

        if d_pan == 0.0 and d_tilt == 0.0:
            return

        target = {n: self.joint_pos[n] for n in JOINT_NAMES}
        target['lbr_A1'] = self._clamp('lbr_A1', target['lbr_A1'] + d_pan)
        target['lbr_A6'] = self._clamp('lbr_A6', target['lbr_A6'] + d_tilt)

        traj = JointTrajectory()
        traj.joint_names = JOINT_NAMES
        pt = JointTrajectoryPoint()
        pt.positions = [target[n] for n in JOINT_NAMES]
        pt.time_from_start = Duration(sec=0, nanosec=200_000_000)
        traj.points = [pt]
        self.traj_pub.publish(traj)

    def _clamp(self, joint, value):
        lim = LIMITS.get(joint)
        return max(-lim, min(lim, value)) if lim else value


def main(args=None):
    rclpy.init(args=args)
    node = FaceTrackerController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
