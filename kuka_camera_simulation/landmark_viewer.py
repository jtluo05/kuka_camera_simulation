#!/usr/bin/env python3
"""Shows landmark index numbers on the face so we can identify forehead points."""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import os
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from ament_index_python.packages import get_package_share_directory

class LandmarkViewer(Node):
    def __init__(self):
        super().__init__('landmark_viewer')
        model_path = os.path.join(
            get_package_share_directory('kuka_camera_simulation'),
            'models', 'mediapipe', 'face_landmarker.task')
        options = mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        self.bridge = CvBridge()
        self.last_ts = -1
        self.pub = self.create_publisher(Image, '/landmark_debug', 10)
        self.create_subscription(Image, '/camera/camera/color/image_raw',
                                 self.cb, 10)
        self.get_logger().info('Landmark viewer running - check /landmark_debug')

    def cb(self, msg):
        ts_ms = msg.header.stamp.sec * 1000 + msg.header.stamp.nanosec // 1_000_000
        if ts_ms <= self.last_ts:
            return
        self.last_ts = ts_ms
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        result = self.landmarker.detect_for_video(mp_image, ts_ms)
        if not result.face_landmarks:
            return
        lm = result.face_landmarks[0]
        h, w = bgr.shape[:2]
        # Draw only landmarks in the upper half of the face (forehead area)
        for i, pt in enumerate(lm):
            if pt.y < 0.55:  # upper 55% of image = forehead area
                px = int(pt.x * w)
                py = int(pt.y * h)
                cv2.circle(bgr, (px, py), 3, (0, 255, 0), -1)
                # Show index number
                cv2.putText(bgr, str(i), (px+3, py-3),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 0), 1)
        out = self.bridge.cv2_to_imgmsg(
            cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), encoding='rgb8')
        self.pub.publish(out)

def main(args=None):
    rclpy.init(args=args)
    node = LandmarkViewer()
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
