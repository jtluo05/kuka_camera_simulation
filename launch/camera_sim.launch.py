from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch_ros.parameter_descriptions import ParameterValue


def launch_setup(context, *args, **kwargs):
    robot_name = LaunchConfiguration("robot_name").perform(context)
    camera = LaunchConfiguration("camera").perform(context)

    if camera == "realsense":
        urdf_file = "lbr_with_realsense.xacro"
    else:
        urdf_file = "lbr_with_camera.xacro"

    robot_description = ParameterValue(
        Command([
            FindExecutable(name="xacro"),
            " ",
            FindPackageShare("kuka_camera_simulation"),
            "/urdf/",
            urdf_file,
            " robot_name:=",
            robot_name,
            " mode:=gazebo",
        ]),
        value_type=str,
    )

    nodes = [
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            output="screen",
            parameters=[
                {"robot_description": robot_description},
                {"use_sim_time": True},
            ],
            namespace=robot_name,
        ),
        IncludeLaunchDescription(
            FindPackageShare("ros_gz_sim") / "launch" / "gz_sim.launch.py",
            launch_arguments={
                "gz_args": [
                    "-r ",
                    PathJoinSubstitution([
                        FindPackageShare("kuka_camera_simulation"),
                        "worlds",
                        "camera_test.world",
                    ]),
                ]
            }.items(),
        ),
        Node(
            package="ros_gz_bridge",
            executable="parameter_bridge",
            name="clock_bridge",
            arguments=["/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock"],
            output="screen",
        ),
        Node(
            package="ros_gz_sim",
            executable="create",
            arguments=[
                "-topic", "robot_description",
                "-name", robot_name,
                "-allow_renaming",
                "-x", "0.0", "-y", "0.0", "-z", "0.0",
            ],
            output="screen",
            namespace=robot_name,
        ),
        Node(
            package="ros_gz_sim",
            executable="create",
            arguments=[
                "-world", "empty",
                "-file", PathJoinSubstitution([
                    FindPackageShare("kuka_camera_simulation"),
                    "models", "db_face", "model.sdf",
                ]),
                "-name", "db_face",
                "-x", "-0.8", "-y", "0", "-z", "1.15",
                "-R", "0", "-P", "0.1745", "-Y", "0",
            ],
            output="screen",
        ),
        Node(
            package="ros_gz_sim",
            executable="create",
            arguments=[
                "-world", "empty",
                "-file", PathJoinSubstitution([
                    FindPackageShare("kuka_camera_simulation"),
                    "models", "laser_dot", "model.sdf",
                ]),
                "-name", "laser_dot",
                "-x", "0", "-y", "0", "-z", "-5.0",
            ],
            output="screen",
        ),
        Node(
            package="controller_manager",
            executable="spawner",
            output="screen",
            arguments=[
                "--controller-manager", "controller_manager",
                "joint_state_broadcaster",
                "joint_trajectory_controller",
            ],
            namespace=robot_name,
        ),
    ]

    if camera == "realsense":
        # Gazebo natively publishes hardware-style topic names
        # (confirmed via: gz topic -l). Bridge them 1:1 into ROS.
        nodes.append(Node(
            package="ros_gz_bridge",
            executable="parameter_bridge",
            name="camera_color_bridge",
            arguments=[
                "/camera/color/image_raw@sensor_msgs/msg/Image[gz.msgs.Image",
            ],
            output="screen",
        ))
        nodes.append(Node(
            package="ros_gz_bridge",
            executable="parameter_bridge",
            name="camera_depth_bridge",
            arguments=[
                "/camera/depth/image_rect_raw@sensor_msgs/msg/Image[gz.msgs.Image",
            ],
            output="screen",
        ))
        nodes.append(Node(
            package="ros_gz_bridge",
            executable="parameter_bridge",
            name="camera_info_bridge",
            arguments=[
                "/camera/color/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo",
            ],
            output="screen",
        ))
    else:
        nodes.append(Node(
            package="ros_gz_bridge",
            executable="parameter_bridge",
            name="camera_bridge",
            arguments=[
                "/camera/image_raw@sensor_msgs/msg/Image[gz.msgs.Image",
            ],
            output="screen",
        ))

    return nodes


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            name="robot_name",
            default_value="lbr",
            description="Robot name / namespace.",
        ),
        DeclareLaunchArgument(
            name="camera",
            default_value="simple",
            description="Camera type: simple or realsense.",
        ),
        OpaqueFunction(function=launch_setup),
    ])
