"""Raster sweep executor v2 (B3+B4+C3).

C3: waypoints are now face-relative. The sweep tracks the face's live
world position (from /face_tracking/world_position) and shifts every
waypoint by how far the face has moved from its reference position.
Face drifts -> targets drift with it.
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Point
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from std_msgs.msg import String

import tf2_ros

from kuka_camera_simulation.raster_planner import ZONES, generate_raster

JOINT_NAMES = ['lbr_A1', 'lbr_A2', 'lbr_A3', 'lbr_A4',
               'lbr_A5', 'lbr_A6', 'lbr_A7']
LIMITS = {'lbr_A1': 2.9, 'lbr_A6': 2.0}

FACE_X = -0.8
# Face position the zones were calibrated against (spawn position)
REF_FACE_Y = 0.0
REF_FACE_Z = 1.15

FOCAL_PX = 553.8
IMG_W, IMG_H = 640, 480


class RasterSweepNode(Node):
    def __init__(self):
        super().__init__('raster_sweep')

        self.declare_parameter('zone', 'forehead')
        self.declare_parameter('line_spacing', 0.01)
        self.declare_parameter('point_spacing', 0.01)
        self.declare_parameter('dwell_time', 0.4)
        self.declare_parameter('aim_threshold_px', 8.0)
        self.declare_parameter('pan_gain', -0.8)
        self.declare_parameter('tilt_gain', -0.8)
        self.declare_parameter('max_step', 0.03)
        self.declare_parameter('rate_hz', 10.0)
        # C3: if no face position arrives, fall back to fixed-world mode
        self.declare_parameter('require_face', True)
        # C2: movement pause/resume
        self.declare_parameter('movement_threshold', 0.001)  # m, pause trigger
        self.declare_parameter('settle_tolerance', 0.005)   # m, "still" band
        self.declare_parameter('settle_frames', 15)         # ticks of stillness

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

        # C3 state: live face offset from the reference position
        self.face_offset_y = 0.0
        self.face_offset_z = 0.0
        self.have_face = False
        # C2 state
        self.paused = False
        self.wp_start_offset = None      # face offset when waypoint began
        self.settle_ref = None           # face offset reference while settling
        self.settle_count = 0

        self.joint_pos = {}
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.create_subscription(JointState, '/lbr/joint_states',
                                 self.joints_cb, 10)
        self.create_subscription(Point, '/face_tracking/world_position',
                                 self.face_cb, 10)
        self.traj_pub = self.create_publisher(
            JointTrajectory,
            '/lbr/joint_trajectory_controller/joint_trajectory', 10)
        self.progress_pub = self.create_publisher(
            String, '/raster_sweep/progress', 10)

        rate = self.get_parameter('rate_hz').value
        self.timer = self.create_timer(1.0 / rate, self.tick)
        self.get_logger().info(
            f"Sweeping zone '{zone_name}': {len(self.waypoints)} waypoints "
            f"(face-relative mode)")

    def joints_cb(self, msg):
        for name, pos in zip(msg.name, msg.position):
            self.joint_pos[name] = pos

    def face_cb(self, msg):
        # How far has the face moved from where the zones were calibrated?
        self.face_offset_y = msg.y - REF_FACE_Y
        self.face_offset_z = msg.z - REF_FACE_Z
        self.have_face = True

    def current_target(self):
        """C3: waypoint shifted by the face's current offset."""
        wy, wz = self.waypoints[self.wp_index]
        return (wy + self.face_offset_y, wz + self.face_offset_z)

    def project_target(self, ty, tz):
        """World point on the face plane -> pixel error from image center."""
        try:
            tfm = self.tf_buffer.lookup_transform(
                'world', 'realsense_link', rclpy.time.Time())
        except Exception:
            return None

        t = tfm.transform.translation
        q = tfm.transform.rotation

        wx = FACE_X - t.x
        wy = ty - t.y
        wz = tz - t.z

        qx, qy, qz, qw = -q.x, -q.y, -q.z, q.w
        ix = qw * wx + qy * wz - qz * wy
        iy = qw * wy + qz * wx - qx * wz
        iz = qw * wz + qx * wy - qy * wx
        iw = -qx * wx - qy * wy - qz * wz
        cx = ix * qw + iw * -qx + iy * -qz - iz * -qy
        cy = iy * qw + iw * -qy + iz * -qx - ix * -qz
        cz = iz * qw + iw * -qz + ix * -qy - iy * -qx

        if cx <= 0.05:
            return None
        px = FOCAL_PX * (-cy) / cx
        py = FOCAL_PX * (-cz) / cx
        return (px, py)

    def tick(self):
        if self.done or len(self.joint_pos) < len(JOINT_NAMES):
            return
        if self.get_parameter('require_face').value and not self.have_face:
            return  # wait until the detector reports a face position

        now = self.get_clock().now()

        # ---- C2: movement pause / settle / resume ----
        cur = (self.face_offset_y, self.face_offset_z)
        if self.wp_start_offset is None:
            self.wp_start_offset = cur

        if not self.paused:
            moved = max(abs(cur[0] - self.wp_start_offset[0]),
                        abs(cur[1] - self.wp_start_offset[1]))
            if moved > self.get_parameter('movement_threshold').value:
                self.paused = True
                self.settle_ref = cur
                self.settle_count = 0
                self.dwell_until = None
                self.get_logger().info(
                    'Face moved %.1f cm - sweep PAUSED, waiting to settle'
                    % (moved * 100))
                self.progress_pub.publish(String(data='paused'))
                return
        else:
            still = max(abs(cur[0] - self.settle_ref[0]),
                        abs(cur[1] - self.settle_ref[1]))
            if still <= self.get_parameter('settle_tolerance').value:
                self.settle_count += 1
            else:
                self.settle_ref = cur
                self.settle_count = 0
            if self.settle_count >= int(self.get_parameter('settle_frames').value):
                self.paused = False
                self.wp_start_offset = cur
                self.get_logger().info(
                    'Face settled - RESUMING at waypoint %d/%d'
                    % (self.wp_index + 1, len(self.waypoints)))
                self.progress_pub.publish(String(data='resumed'))
            return  # while paused/settling: no motion commands at all

        if self.dwell_until is not None:
            if now < self.dwell_until:
                return
            self.dwell_until = None
            self.wp_index += 1
            self.wp_start_offset = (self.face_offset_y, self.face_offset_z)
            if self.wp_index >= len(self.waypoints):
                self.get_logger().info('Sweep complete.')
                self.progress_pub.publish(String(data='complete'))
                self.done = True
                return

        ty, tz = self.current_target()
        err = self.project_target(ty, tz)
        if err is None:
            return
        dx, dy = err

        thresh = self.get_parameter('aim_threshold_px').value
        if abs(dx) < thresh and abs(dy) < thresh:
            dwell = self.get_parameter('dwell_time').value
            self.dwell_until = now + rclpy.duration.Duration(seconds=dwell)
            msg = f'waypoint {self.wp_index + 1}/{len(self.waypoints)}'
            self.progress_pub.publish(String(data=msg))
            if (self.wp_index + 1) % 10 == 0 or self.wp_index == 0:
                self.get_logger().info(msg)
            return

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
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
