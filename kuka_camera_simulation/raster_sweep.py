"""Raster sweep executor (B3+B4).

Aims the laser at each planned waypoint in turn by servoing the arm —
the same nudge-until-centered loop as the face tracker, but the target
is a projected waypoint instead of a detected nose.

Usage:
    ros2 run kuka_camera_simulation raster_sweep
    ros2 run kuka_camera_simulation raster_sweep --ros-args -p zone:=left_cheek
"""
import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from std_msgs.msg import String

import tf2_ros

from kuka_camera_simulation.raster_planner import ZONES, generate_raster

JOINT_NAMES = ['lbr_A1', 'lbr_A2', 'lbr_A3', 'lbr_A4',
               'lbr_A5', 'lbr_A6', 'lbr_A7']
LIMITS = {'lbr_A1': 2.9, 'lbr_A6': 2.0}

# Face plane X position (models the treated surface)
FACE_X = -0.8
# Camera intrinsics: 640px wide, 1.047 rad horizontal FOV
FOCAL_PX = 553.8
IMG_W, IMG_H = 640, 480


class RasterSweepNode(Node):
    def __init__(self):
        super().__init__('raster_sweep')

        self.declare_parameter('zone', 'forehead')
        self.declare_parameter('line_spacing', 0.01)
        self.declare_parameter('point_spacing', 0.01)
        self.declare_parameter('dwell_time', 0.4)     # s at each point
        self.declare_parameter('aim_threshold_px', 8.0)
        self.declare_parameter('pan_gain', -0.8)
        self.declare_parameter('tilt_gain', -0.8)
        self.declare_parameter('max_step', 0.03)
        self.declare_parameter('rate_hz', 10.0)

        zone_name = self.get_parameter('zone').value
        if zone_name not in ZONES:
            self.get_logger().error(
                f"Unknown zone '{zone_name}'. Options: {list(ZONES)}")
            raise SystemExit(1)

        self.waypoints = generate_raster(
            ZONES[zone_name],
            self.get_parameter('line_spacing').value,
            self.get_parameter('point_spacing').value)
        self.wp_index = 0
        self.dwell_until = None
        self.done = False

        self.joint_pos = {}
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.create_subscription(JointState, '/lbr/joint_states',
                                 self.joints_cb, 10)
        self.traj_pub = self.create_publisher(
            JointTrajectory,
            '/lbr/joint_trajectory_controller/joint_trajectory', 10)
        self.progress_pub = self.create_publisher(
            String, '/raster_sweep/progress', 10)

        rate = self.get_parameter('rate_hz').value
        self.timer = self.create_timer(1.0 / rate, self.tick)
        self.get_logger().info(
            f"Sweeping zone '{zone_name}': {len(self.waypoints)} waypoints")

    def joints_cb(self, msg):
        for name, pos in zip(msg.name, msg.position):
            self.joint_pos[name] = pos

    def project_waypoint(self, wp):
        """Where does waypoint (y, z) on the face plane appear in the
        camera image? Returns pixel error (dx, dy) from image center,
        or None if TF isn't ready."""
        try:
            tfm = self.tf_buffer.lookup_transform(
                'world', 'realsense_link', rclpy.time.Time())
        except Exception:
            return None

        t = tfm.transform.translation
        q = tfm.transform.rotation

        # world-space vector from camera to the waypoint
        wx = FACE_X - t.x
        wy = wp[0] - t.y
        wz = wp[1] - t.z

        # rotate into camera frame: apply inverse quaternion
        # (conjugate for unit quaternion)
        qx, qy, qz, qw = -q.x, -q.y, -q.z, q.w
        # v' = q * v * q_conj  (expanded)
        ix = qw * wx + qy * wz - qz * wy
        iy = qw * wy + qz * wx - qx * wz
        iz = qw * wz + qx * wy - qy * wx
        iw = -qx * wx - qy * wy - qz * wz
        cx = ix * qw + iw * -qx + iy * -qz - iz * -qy
        cy = iy * qw + iw * -qy + iz * -qx - ix * -qz
        cz = iz * qw + iw * -qz + ix * -qy - iy * -qx

        # camera looks along its local +X; image x right = -Y, image y down = -Z
        if cx <= 0.05:
            return None  # behind or too close
        px = FOCAL_PX * (-cy) / cx
        py = FOCAL_PX * (-cz) / cx
        return (px, py)

    def tick(self):
        if self.done or len(self.joint_pos) < len(JOINT_NAMES):
            return

        now = self.get_clock().now()

        # dwelling at a converged waypoint?
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

        wp = self.waypoints[self.wp_index]
        err = self.project_waypoint(wp)
        if err is None:
            return
        dx, dy = err

        thresh = self.get_parameter('aim_threshold_px').value
        if abs(dx) < thresh and abs(dy) < thresh:
            # aimed: start dwell (the pretend laser shot)
            dwell = self.get_parameter('dwell_time').value
            self.dwell_until = now + rclpy.duration.Duration(seconds=dwell)
            msg = f'waypoint {self.wp_index + 1}/{len(self.waypoints)}'
            self.progress_pub.publish(String(data=msg))
            if (self.wp_index + 1) % 10 == 0 or self.wp_index == 0:
                self.get_logger().info(msg)
            return

        # servo toward the waypoint: normalized error -> joint nudges
        nx = dx / IMG_W
        ny = dy / IMG_H
        max_step = self.get_parameter('max_step').value
        d_pan = max(-max_step, min(max_step,
                    self.get_parameter('pan_gain').value * nx))
        d_tilt = max(-max_step, min(max_step,
                     self.get_parameter('tilt_gain').value * ny))

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
    node = RasterSweepNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
