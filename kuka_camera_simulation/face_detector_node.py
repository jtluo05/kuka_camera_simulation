"""Face detection node: finds a face in the camera stream and publishes
how far off-center it is, for the tracking controller to consume.

Subscribes:  /camera/image_raw          (sensor_msgs/Image)
Publishes:   /face_tracking/error       (geometry_msgs/Point)
                 x: horizontal offset of nose tip from image center,
                    normalized [-0.5..0.5], positive = face is RIGHT of center
                 y: vertical offset, positive = face is BELOW center
                 z: 1.0 if a face is detected this frame, 0.0 if not
             /face_tracking/debug_image (sensor_msgs/Image)
                 camera frame with landmarks + crosshair drawn on it
"""
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
import os

NOSE_TIP_INDEX = 1  # index of the nose tip in the 478-landmark face mesh


class FaceDetectorNode(Node):
    def __init__(self):
        super().__init__('face_detector_node')

        # Tunable via: ros2 run ... --ros-args -p min_detection_confidence:=0.3
        self.declare_parameter('min_detection_confidence', 0.5)
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

        self.error_pub = self.create_publisher(Point, '/face_tracking/error', 10)
        self.debug_pub = self.create_publisher(Image, '/face_tracking/debug_image', 10)
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10)

        self.get_logger().info(f'Face detector running (model: {model_path})')

    def image_callback(self, msg: Image):
        # VIDEO mode needs strictly increasing timestamps in ms; use sim time
        ts_ms = msg.header.stamp.sec * 1000 + msg.header.stamp.nanosec // 1_000_000
        if ts_ms <= self.last_ts_ms:
            return  # duplicate/old frame, skip
        self.last_ts_ms = ts_ms

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        result = self.landmarker.detect_for_video(mp_image, ts_ms)

        h, w = frame.shape[:2]
        error = Point()

        if result.face_landmarks:
            landmarks = result.face_landmarks[0]
            nose = landmarks[NOSE_TIP_INDEX]
            # landmark coords are normalized [0..1]; center of image is (0.5, 0.5)
            error.x = nose.x - 0.5
            error.y = nose.y - 0.5
            error.z = 1.0

            # draw all landmarks small, nose tip big
            for lm in landmarks:
                cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 1, (0, 255, 0), -1)
            cv2.circle(frame, (int(nose.x * w), int(nose.y * h)), 5, (255, 0, 0), -1)
        else:
            error.x = 0.0
            error.y = 0.0
            error.z = 0.0

        # crosshair at image center
        cv2.line(frame, (w // 2 - 15, h // 2), (w // 2 + 15, h // 2), (255, 255, 0), 1)
        cv2.line(frame, (w // 2, h // 2 - 15), (w // 2, h // 2 + 15), (255, 255, 0), 1)

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
        rclpy.shutdown()


if __name__ == '__main__':
    main()
