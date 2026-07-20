from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch_ros.parameter_descriptions import ParameterValue
def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                name="robot_name",
                default_value="lbr",
                description="Robot name / namespace.",
            ),
            # Build the combined arm+camera robot description from our xacro
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                output="screen",
                parameters=[
                    {
                        "robot_description": ParameterValue(
                            Command(
                            [
                                FindExecutable(name="xacro"),
                                " ",
                                FindPackageShare("kuka_camera_simulation"),
                                "/urdf/lbr_with_camera.xacro",
                                " robot_name:=",
                                LaunchConfiguration("robot_name"),
                                " mode:=gazebo",
                            ]
                            ),
                            value_type=str,
                        )
                    },
                    {"use_sim_time": True},
                ],
                namespace=LaunchConfiguration("robot_name"),
            ),
            # Start Gazebo (empty world)
            IncludeLaunchDescription(
                FindPackageShare("ros_gz_sim") / "launch" / "gz_sim.launch.py",
                launch_arguments={
                    "gz_args": [
                        "-r ",
                        PathJoinSubstitution(
                            [
                                FindPackageShare("kuka_camera_simulation"),
                                "worlds",
                                "camera_test.world",
                            ]
                        ),
                    ]
                }.items(),
            ),
            # Bridge simulation clock
            Node(
                package="ros_gz_bridge",
                executable="parameter_bridge",
                name="clock_bridge",
                arguments=["/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock"],
                output="screen",
            ),
            # Bridge the camera image topic from Gazebo into ROS 2
            Node(
                package="ros_gz_bridge",
                executable="parameter_bridge",
                name="camera_bridge",
                arguments=[
                    "/camera/image_raw@sensor_msgs/msg/Image[gz.msgs.Image"
                ],
                output="screen",
            ),
            # Spawn the combined model in Gazebo
            Node(
                package="ros_gz_sim",
                executable="create",
                arguments=[
                    "-topic", "robot_description",
                    "-name", LaunchConfiguration("robot_name"),
                    "-allow_renaming",
                    "-x", "0.0", "-y", "0.0", "-z", "0.0",
                ],
                output="screen",
                namespace=LaunchConfiguration("robot_name"),
            ),
            # Spawn the scanned face in front of the camera
            Node(
                package="ros_gz_sim",
                executable="create",
                arguments=[
                    "-world", "empty",
                    "-file", PathJoinSubstitution(
                        [
                            FindPackageShare("kuka_camera_simulation"),
                            "models", "db_face", "model.sdf",
                        ]
                    ),
                    "-name", "db_face",
                    "-x", "-0.9", "-y", "0", "-z", "1.2", "-R", "0", "-P", "0.1745", "-Y", "0",
                ],
                output="screen",
            ),
            # Load controllers
            Node(
                package="controller_manager",
                executable="spawner",
                output="screen",
                arguments=[
                    "--controller-manager", "controller_manager",
                    "joint_state_broadcaster",
                    "joint_trajectory_controller",
                ],
                namespace=LaunchConfiguration("robot_name"),
            ),
        ]
    )
