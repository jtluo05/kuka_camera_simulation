"""Raster sweep v3 - pixel offset based.
No TF, no world coordinates. Works purely in image pixel space.
Offsets are relative to the nose tip position in the image.
Uses same control loop as face_tracker_controller which is proven working.

Run with:
ros2 run kuka_camera_simulation raster_sweep_pixel
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Point
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from std_msgs.msg import String

JOINT_NAMES = ['lbr_A1', 'lbr_A2', 'lbr_A3', 'lbr_A4',
               'lbr_A5', 'lbr_A6', 'lbr_A7']
LIMITS = {'lbr_A1': 2.9, 'lbr_A6': 2.0}
IMG_W, IMG_H = 1280, 720


def generate_pixel_sweep(x_min, x_max, y_min, y_max, x_step, y_step):
    """S-pattern in pixel space relative to nose."""
    waypoints = []
    y = y_min
    row = 0
    while y <= y_max:
        xs = list(range(x_min, x_max + 1, x_step))
        if row % 2 == 1:
            xs = xs[::-1]
        for x in xs:
            waypoints.append((x, y))
        y += y_step
        row += 1
    return waypoints


class RasterSweepPixelNode(Node):
    def __init__(self):
        super().__init__('raster_sweep_pixel')

        self.declare_parameter('pan_gain',  -0.3)
        self.declare_parameter('tilt_gain',  0.3)
        self.declare_parameter('max_step',   0.008)
        self.declare_parameter('rate_hz',    10.0)
        self.declare_parameter('dwell_time', 0.5)
        self.declare_parameter('aim_threshold_px', 20.0)
        # Forehead sweep area in pixels relative to nose tip
        # Negative y = above nose, positive y = below nose
        self.declare_parameter('x_min', -150)   # pixels left of nose
        self.declare_parameter('x_max',  150)   # pixels right of nose
        self.declare_parameter('y_min', -200)   # pixels above nose (forehead)
        self.declare_parameter('y_max',  -80)   # bottom of forehead zone
        self.declare_parameter('x_step',  30)   # horizontal spacing
        self.declare_parameter('y_step',  30)   # row spacing

        self.waypoints = generate_pixel_sweep(
            x_min=self.get_parameter('x_min').value,
            x_max=self.get_parameter('x_max').value,
            y_min=self.get_parameter('y_min').value,
            y_max=self.get_parameter('y_max').value,
            x_step=self.get_parameter('x_step').value,
            y_step=self.get_parameter('y_step').value,
        )
        self.wp_index = 0
        self.dwell_until = None
        self.done = False
        self.nose_x = 0.0
        self.nose_y = 0.0
        self.have_face = False
        self.joint_pos = {}

        self.create_subscription(JointState, '/lbr/joint_states',
                                 self.joints_cb, 10)
        self.create_subscription(Point, '/face_tracking/error',
                                 self.error_cb, 10)
        self.traj_pub = self.create_publisher(
            JointTrajectory,
            '/lbr/joint_trajectory_controller/joint_trajectory', 10)
        self.progress_pub = self.create_publisher(
            String, '/raster_sweep/progress', 10)

        rate = self.get_parameter('rate_hz').value
        self.timer = self.create_timer(1.0 / rate, self.tick)
        self.get_logger().info(
            f'Pixel sweep ready: {len(self.waypoints)} waypoints. ')
        self.get_logger().info(
            f'Forehead zone: x=[{self.get_parameter("x_min").value}, ' 
            f'{self.get_parameter("x_max").value}]px, ' 
            f'y=[{self.get_parameter("y_min").value}, ' 
            f'{self.get_parameter("y_max").value}]px from nose')

    def joints_cb(self, msg):
        for name, pos in zip(msg.name, msg.position):
            self.joint_pos[name] = pos

    def error_cb(self, msg):
        if msg.z < 0.5:
            return
        self.nose_x = msg.x  # normalized -0.5 to 0.5
        self.nose_y = msg.y
        self.have_face = True

    def tick(self):
        if self.done or not self.have_face:
            return
        if len(self.joint_pos) < len(JOINT_NAMES):
            return

        now = self.get_clock().now()

        if self.dwell_until is not None:
            if now < self.dwell_until:
                return
            self.dwell_until = None
            self.wp_index += 1
            if self.wp_index >= len(self.waypoints):
                self.get_logger().info('Sweep complete.')
                self.progress_pub.publish(String(data='complete'))
                self.done = True
                return

        # Waypoint offset in pixels from nose
        wp_px_x, wp_px_y = self.waypoints[self.wp_index]

        # Nose position in pixels from image center
        nose_px_x = self.nose_x * IMG_W
        nose_px_y = self.nose_y * IMG_H

        # Target pixel = nose position + offset
        target_px_x = nose_px_x + wp_px_x
        target_px_y = nose_px_y + wp_px_y

        # Normalized error from image center
        dx = target_px_x / IMG_W
        dy = target_px_y / IMG_H

        thresh = self.get_parameter('aim_threshold_px').value / IMG_W
        if abs(dx) < thresh and abs(dy) < thresh:
            dwell = self.get_parameter('dwell_time').value
            self.dwell_until = now + rclpy.duration.Duration(seconds=dwell)
            msg = f'waypoint {self.wp_index + 1}/{len(self.waypoints)}'
            self.progress_pub.publish(String(data=msg))
            if (self.wp_index + 1) % 5 == 0 or self.wp_index == 0:
                self.get_logger().info(msg)
            return

        max_step = self.get_parameter('max_step').value
        d_pan  = max(-max_step, min(max_step,
                     self.get_parameter('pan_gain').value  * dx))
        d_tilt = max(-max_step, min(max_step,
                     self.get_parameter('tilt_gain').value * dy))

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
    node = RasterSweepPixelNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
