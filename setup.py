import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'kuka_camera_simulation'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/' + package_name + '/models/mediapipe', glob('models/mediapipe/*')),
        ('share/' + package_name + '/models/db_face', ['models/db_face/model.sdf']),
        ('share/' + package_name + '/models/laser_dot', ['models/laser_dot/model.sdf']),
        ('share/' + package_name + '/models/db_face/meshes', glob('models/db_face/meshes/*')),
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*.xacro')),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*.world')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jordanl',
    maintainer_email='jordanl@todo.todo',
    description='Camera simulation package for KUKA LBR iiwa14 end effector',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'raster_planner = kuka_camera_simulation.raster_planner:main',
            'laser_dot_node = kuka_camera_simulation.laser_dot_node:main',
            'face_tracker_controller = kuka_camera_simulation.face_tracker_controller:main',
            'face_detector_node = kuka_camera_simulation.face_detector_node:main',
        ],
    },
)