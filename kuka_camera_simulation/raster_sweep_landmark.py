"""Raster sweep v5 - uses actual MediaPipe landmark positions as waypoints.
The forehead landmarks ARE the waypoints - no zone computation needed.
Scales perfectly with any face at any distance or angle.
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Image
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from std_msgs.msg import String
from cv_bridge import CvBridge
import numpy as np
import os

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from ament_index_python.packages import get_package_share_directory

JOINT_NAMES = ['lbr_A1', 'lbr_A2', 'lbr_A3', 'lbr_A4',
               'lbr_A5', 'lbr_A6', 'lbr_A7']
LIMITS = {'lbr_A1': 2.9, 'lbr_A6': 2.0}
IMG_W, IMG_H = 1280, 720

# Forehead landmark indices - verified against MediaPipe face mesh map
# Row 1: hairline/top forehead, right to left
# Row 2: mid forehead, left to right (S-pattern reversal)  
# Row 3: lower forehead above eyebrows, right to left
# Note: MediaPipe x-axis - lower index = left side of face in image
FOREHEAD_LANDMARKS = [
    # Row 1 - right to left (added 251 at end for full coverage)
    [54, 103, 67, 109, 10, 338, 297, 332, 284, 251],
    # Row 2 - left to right (added 162 at end for full coverage)
    [251, 298, 333, 299, 337, 151, 108, 69, 104, 68, 21, 162],
    # Row 3 - right to left (added 356 at end for full coverage)
    [162, 71, 63, 105, 66, 107, 9, 336, 296, 334, 293, 301, 389, 356],
]

NOSE_TIP = 1


class RasterSweepLandmarkNode(Node):
    def __init__(self):
        super().__init__('raster_sweep_landmark')

        self.declare_parameter('pan_gain',  -0.3)
        self.declare_parameter('tilt_gain',  0.3)
        self.declare_parameter('max_step',   0.015)
        self.declare_parameter('rate_hz',    10.0)
        self.declare_parameter('dwell_time', 0.05)
        self.declare_parameter('aim_threshold_px', 20.0)
        # Lens offset bias - same as face tracker (camera lens not at center)
        self.declare_parameter('x_bias', -0.053)  # negative = shift aim left
        self.declare_parameter('y_bias', 0.0)
        self.declare_parameter('image_topic',
                               '/camera/camera/color/image_raw')

        model_path = os.path.join(
            get_package_share_directory('kuka_camera_simulation'),
            'models', 'mediapipe', 'face_landmarker.task')
        options = mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=model_path),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        self.bridge = CvBridge()
        self.last_ts_ms = -1

        # Waypoints as pixel offsets from nose, updated each frame
        self.waypoints = []
        self.wp_index = 0
        self.dwell_until = None
        self.done = False
        self.have_face = False
        self.nose_x = 0.0  # normalized
        self.nose_y = 0.0
        self.joint_pos = {}
        self.zone_set = False

        self.create_subscription(JointState, '/lbr/joint_states',
                                 self.joints_cb, 10)
        self.create_subscription(
            Image,
            self.get_parameter('image_topic').value,
            self.image_cb, 10)
        self.traj_pub = self.create_publisher(
            JointTrajectory,
            '/lbr/joint_trajectory_controller/joint_trajectory', 10)
        self.progress_pub = self.create_publisher(
            String, '/raster_sweep/progress', 10)

        rate = self.get_parameter('rate_hz').value
        self.timer = self.create_timer(1.0 / rate, self.tick)
        self.get_logger().info('Landmark sweep v5 ready - waiting for face')

    def joints_cb(self, msg):
        for name, pos in zip(msg.name, msg.position):
            self.joint_pos[name] = pos

    def image_cb(self, msg):
        ts_ms = (msg.header.stamp.sec * 1000
                 + msg.header.stamp.nanosec // 1_000_000)
        if ts_ms <= self.last_ts_ms:
            return
        self.last_ts_ms = ts_ms

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        result = self.landmarker.detect_for_video(mp_image, ts_ms)
        if not result.face_landmarks:
            return

        lm = result.face_landmarks[0]
        h, w = frame.shape[:2]

        # Nose position
        nose = lm[NOSE_TIP]
        nose_px_x = nose.x * w
        nose_px_y = nose.y * h
        self.nose_x = nose.x - 0.5
        self.nose_y = nose.y - 0.5
        self.have_face = True

        # Compute waypoints from actual landmark positions
        # Only compute once at start (or after completion)
        if not self.zone_set:
            waypoints = []
            for row_idx, row in enumerate(FOREHEAD_LANDMARKS):
                row_pts = []
                for lm_idx in row:
                    if lm_idx < len(lm):
                        pt = lm[lm_idx]
                        # Offset from nose in pixels
                        dx = int(pt.x * w - nose_px_x)
                        dy = int(pt.y * h - nose_px_y)
                        row_pts.append((dx, dy))
                # Rows already in correct S-pattern order in FOREHEAD_LANDMARKS
                # No reversal needed - order defined in the landmark lists above
                pass
                waypoints.extend(row_pts)

            self.waypoints = waypoints
            self.wp_index = 0
            self.done = False
            self.zone_set = True
            self.get_logger().info(
                f'Zone set from {len(self.waypoints)} forehead landmarks')

    def tick(self):
        if self.done or not self.have_face or not self.zone_set:
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

        wp_px_x, wp_px_y = self.waypoints[self.wp_index]

        # Target = current nose position + landmark offset
        nose_px_x = self.nose_x * IMG_W
        nose_px_y = self.nose_y * IMG_H
        target_px_x = nose_px_x + wp_px_x
        target_px_y = nose_px_y + wp_px_y

        # Apply lens offset bias (same correction as face tracker)
        x_bias = self.get_parameter('x_bias').value
        y_bias = self.get_parameter('y_bias').value
        dx = target_px_x / IMG_W + x_bias
        dy = target_px_y / IMG_H + y_bias

        thresh = self.get_parameter('aim_threshold_px').value / IMG_W
        if abs(dx) < thresh and abs(dy) < thresh:
            dwell = self.get_parameter('dwell_time').value
            self.dwell_until = now + rclpy.duration.Duration(seconds=dwell)
            msg = f'waypoint {self.wp_index + 1}/{len(self.waypoints)}'
            self.progress_pub.publish(String(data=msg))
            if (self.wp_index + 1) % 4 == 0 or self.wp_index == 0:
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
    node = RasterSweepLandmarkNode()
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
