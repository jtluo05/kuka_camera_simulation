"""Laser dot projector v2.

Fixes the self-measurement drift: samples depth OFFSET from the dot's
own pixel location, and only moves the dot on meaningful depth changes.
"""
import subprocess

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import tf2_ros


class LaserDotNode(Node):
    def __init__(self):
        super().__init__('laser_dot_node')

        self.declare_parameter('dot_update_hz', 10.0)
        self.declare_parameter('max_range', 5.0)
        # sample this many px to the SIDE of center so we never
        # measure the dot itself
        self.declare_parameter('sample_offset_px', 15)
        # ignore depth changes smaller than this (meters)
        self.declare_parameter('depth_deadband', 0.005)

        self.bridge = CvBridge()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.latest_depth = None
        self.dot_visible = False
        self.last_depth_m = None

        self.create_subscription(
            Image, '/camera/depth/image_rect_raw', self.depth_cb, 10)

        rate = self.get_parameter('dot_update_hz').value
        self.timer = self.create_timer(1.0 / rate, self.update_dot)
        self.get_logger().info('Laser dot node v2 running.')

    def depth_cb(self, msg):
        self.latest_depth = msg

    def move_dot(self, x, y, z):
        result = subprocess.run([
            'gz', 'service', '-s', '/world/empty/set_pose',
            '--reqtype', 'gz.msgs.Pose',
            '--reptype', 'gz.msgs.Boolean',
            '--timeout', '300',
            '--req',
            'name: "laser_dot" position {x: %f y: %f z: %f}' % (x, y, z),
        ], capture_output=True, text=True)
        ok = result.returncode == 0 and 'true' in result.stdout.lower()
        if not ok:
            self.get_logger().warn(
                'set_pose failed: rc=%d out=%s err=%s'
                % (result.returncode, result.stdout.strip(),
                   result.stderr.strip()))
        return ok

    def sample_depth(self, depth, cx, cy):
        """Median of a 5x5 patch at (cx, cy); NaN-safe."""
        h, w = depth.shape[:2]
        cx = max(2, min(w - 3, cx))
        cy = max(2, min(h - 3, cy))
        patch = depth[cy - 2:cy + 3, cx - 2:cx + 3].astype(float)
        finite = patch[np.isfinite(patch)]
        finite = finite[finite > 50]   # 16UC1 in mm, ignore under 50mm
        if finite.size == 0:
            return None
        return float(np.median(finite)) / 1000.0  # mm -> meters

    def update_dot(self):
        if self.latest_depth is None:
            return

        depth = self.bridge.imgmsg_to_cv2(self.latest_depth)
        h, w = depth.shape[:2]
        off = int(self.get_parameter('sample_offset_px').value)

        # Sample LEFT and RIGHT of center; take the closer surface.
        # The dot sits at center, so neither sample sees the dot.
        d_left = self.sample_depth(depth, w // 2 - off, h // 2)
        d_right = self.sample_depth(depth, w // 2 + off, h // 2)
        candidates = [d for d in (d_left, d_right) if d is not None]

        max_range = self.get_parameter('max_range').value
        candidates = [d for d in candidates if d <= max_range]

        if not candidates:
            if self.dot_visible:
                if self.move_dot(0.0, 0.0, -5.0):
                    self.dot_visible = False
                    self.last_depth_m = None
            return

        d = min(candidates)

        # Deadband: skip micro-updates (kills drift and jitter)
        deadband = self.get_parameter('depth_deadband').value
        if (self.dot_visible and self.last_depth_m is not None
                and abs(d - self.last_depth_m) < deadband):
            return

        try:
            tfm = self.tf_buffer.lookup_transform(
                'lbr_link_0', 'realsense_link', rclpy.time.Time())
        except Exception:
            return

        t = tfm.transform.translation
        q = tfm.transform.rotation
        vx = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        vy = 2.0 * (q.x * q.y + q.w * q.z)
        vz = 2.0 * (q.x * q.z - q.w * q.y)

        # place the dot 3mm in FRONT of the surface so it never
        # intersects the face mesh (flush look, no z-fighting)
        d_place = d - 0.003

        x = t.x + d_place * vx
        y = t.y + d_place * vy
        z = t.z + d_place * vz

        if self.move_dot(x, y, z):
            self.dot_visible = True
            self.last_depth_m = d


def main(args=None):
    rclpy.init(args=args)
    node = LaserDotNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
