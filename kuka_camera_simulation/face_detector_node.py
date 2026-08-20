"""Face detection node + world position estimator (C1).

Detects the face with MediaPipe and publishes:
  /face_tracking/error           - normalized nose offset from image center
  /face_tracking/debug_image     - annotated camera frame
  /face_tracking/world_position  - nose tip position in WORLD coordinates
                                   (requires depth; realsense mode only)

World position pipeline:
  nose pixel (MediaPipe) -> depth at that pixel (RealSense)
  -> back-project into camera frame (inverse pinhole)
  -> rotate/translate into world frame (TF)
"""
import os

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from cv_bridge import CvBridge

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from ament_index_python.packages import get_package_share_directory
import tf2_ros

NOSE_TIP_INDEX = 1
FOCAL_PX = 913.0          # 640px wide, 1.047 rad HFOV
IMG_W, IMG_H = 1280, 720


def rotate_vec_by_quat(vx, vy, vz, q):
    """Rotate vector v from the quaternion's child frame into its
    parent frame (camera -> world here)."""
    x, y, z, w = q.x, q.y, q.z, q.w
    # t = 2 * cross(q_xyz, v)
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    # v' = v + w * t + cross(q_xyz, t)
    rx = vx + w * tx + (y * tz - z * ty)
    ry = vy + w * ty + (z * tx - x * tz)
    rz = vz + w * tz + (x * ty - y * tx)
    return rx, ry, rz


class FaceDetectorNode(Node):
    def __init__(self):
        super().__init__('face_detector_node')

        self.declare_parameter('min_detection_confidence', 0.5)
        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('depth_topic', '/camera/depth/image_rect_raw')
        conf = self.get_parameter('min_detection_confidence').value

        model_path = os.path.join(
            get_package_share_directory('kuka_camera_simulation'),
            'models', 'mediapipe', 'face_landmarker.task')

        options = mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=conf,
            min_face_presence_confidence=conf,
            min_tracking_confidence=conf,
        )
        self.landmarker = mp_vision.FaceLandmarker.create_from_options(options)

        self.bridge = CvBridge()
        self.last_ts_ms = -1
        self.latest_depth = None

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        # Smoothing buffer for world position — average over N frames
        # reduces noise from depth sensor fluctuations
        self._world_pos_buffer = []
        self._world_pos_buffer_size = 5

        self.error_pub = self.create_publisher(
            Point, '/face_tracking/error', 10)
        self.debug_pub = self.create_publisher(
            Image, '/face_tracking/debug_image', 10)
        self.world_pub = self.create_publisher(
            Point, '/face_tracking/world_position', 10)

        self.image_sub = self.create_subscription(
            Image, self.get_parameter('image_topic').value,
            self.image_callback, 10)
        self.depth_sub = self.create_subscription(
            Image, self.get_parameter('depth_topic').value,
            self.depth_callback, 10)

        self.get_logger().info(f'Face detector running (model: {model_path})')

    def depth_callback(self, msg):
        self.latest_depth = msg

    def sample_depth_at(self, u, v):
        """Median depth in a 5x5 patch around pixel (u, v); None if no
        depth image or no valid readings."""
        if self.latest_depth is None:
            return None
        depth = self.bridge.imgmsg_to_cv2(self.latest_depth)
        h, w = depth.shape[:2]
        u = max(2, min(w - 3, int(u)))
        v = max(2, min(h - 3, int(v)))
        patch = depth[v - 2:v + 3, u - 2:u + 3].astype(float)
        finite = patch[np.isfinite(patch)]
        # 16UC1 depth is in millimeters — convert to meters
        finite = finite[finite > 50]   # ignore readings under 50mm
        if finite.size == 0:
            return None
        return float(np.median(finite)) / 1000.0  # mm -> meters

    def nose_world_position(self, nose_u, nose_v):
        """Back-project the nose pixel into world coordinates.
        Returns (x, y, z) or None."""
        d = self.sample_depth_at(nose_u, nose_v)
        if d is None:
            return None
        try:
            tfm = self.tf_buffer.lookup_transform(
                'lbr_link_0', 'realsense_link', rclpy.time.Time())
        except Exception:
            return None

        # Inverse pinhole: pixel offsets -> camera-frame vector.
        # Camera looks along local +X; image right = -Y, image down = -Z.
        du = nose_u - IMG_W / 2.0
        dv = nose_v - IMG_H / 2.0
        cx = d
        cy = -du * d / FOCAL_PX
        cz = -dv * d / FOCAL_PX

        # Camera frame -> world frame
        t = tfm.transform.translation
        wx, wy, wz = rotate_vec_by_quat(cx, cy, cz, tfm.transform.rotation)
        return (t.x + wx, t.y + wy, t.z + wz)

    def image_callback(self, msg):
        ts_ms = msg.header.stamp.sec * 1000 + msg.header.stamp.nanosec // 1_000_000
        if ts_ms <= self.last_ts_ms:
            return
        self.last_ts_ms = ts_ms

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        result = self.landmarker.detect_for_video(mp_image, ts_ms)

        h, w = frame.shape[:2]
        error = Point()

        if result.face_landmarks:
            landmarks = result.face_landmarks[0]
            nose = landmarks[NOSE_TIP_INDEX]
            error.x = nose.x - 0.5
            error.y = nose.y - 0.5
            error.z = 1.0

            for lm in landmarks:
                cv2.circle(frame, (int(lm.x * w), int(lm.y * h)),
                           1, (0, 255, 0), -1)
            cv2.circle(frame, (int(nose.x * w), int(nose.y * h)),
                       5, (255, 0, 0), -1)

            # C1: physical world position of the nose tip
            wp = self.nose_world_position(nose.x * w, nose.y * h)
            if wp is not None:
                self._world_pos_buffer.append(wp)
                if len(self._world_pos_buffer) > self._world_pos_buffer_size:
                    self._world_pos_buffer.pop(0)
                # publish smoothed average
                sx = sum(p[0] for p in self._world_pos_buffer) / len(self._world_pos_buffer)
                sy = sum(p[1] for p in self._world_pos_buffer) / len(self._world_pos_buffer)
                sz = sum(p[2] for p in self._world_pos_buffer) / len(self._world_pos_buffer)
                world_msg = Point()
                world_msg.x, world_msg.y, world_msg.z = sx, sy, sz
                self.world_pub.publish(world_msg)
        else:
            error.x = 0.0
            error.y = 0.0
            error.z = 0.0

        cv2.line(frame, (w // 2 - 15, h // 2), (w // 2 + 15, h // 2),
                 (255, 255, 0), 1)
        cv2.line(frame, (w // 2, h // 2 - 15), (w // 2, h // 2 + 15),
                 (255, 255, 0), 1)

        self.error_pub.publish(error)
        debug_msg = self.bridge.cv2_to_imgmsg(frame, encoding='rgb8')
        debug_msg.header = msg.header
        self.debug_pub.publish(debug_msg)


def main(args=None):
    rclpy.init(args=args)
    node = FaceDetectorNode()
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
