"""Raster path planner for facial treatment zones.

B1: zone definitions — rectangles in the face plane (world Y-Z at the
face's X position). B2: S-pattern (boustrophedon) waypoint generation.

Run standalone to preview a zone's waypoints without the robot:
    ros2 run kuka_camera_simulation raster_planner
    ros2 run kuka_camera_simulation raster_planner --ros-args -p zone:=left_cheek
"""
import rclpy
from rclpy.node import Node

# ---------------------------------------------------------------
# B1 — Zone definitions.
# Rectangles in world coordinates on the face plane.
# The face model is centered at y=0, head center near z=1.2.
# Values are initial estimates — calibrate visually with the laser
# dot and adjust here.
#   y: left/right (positive = robot's left / patient's right side)
#   z: up/down
# ---------------------------------------------------------------
ZONES = {
    'forehead': {
        'y_min': -0.064, 'y_max': 0.064,   # ~12.8 cm wide (widened 1.9cm/side)
        'z_min': 1.205, 'z_max': 1.245,    # bottom row removed (eyebrows), face at z=1.15
    },
    'left_cheek': {
        'y_min': 0.025, 'y_max': 0.070,    # patient's left cheek
        'z_min': 1.100, 'z_max': 1.155,
    },
    'right_cheek': {
        'y_min': -0.070, 'y_max': -0.025,
        'z_min': 1.100, 'z_max': 1.155,
    },
}


def generate_raster(zone: dict, line_spacing: float = 0.01,
                    point_spacing: float = 0.01):
    """B2 — Generate S-pattern waypoints for a zone rectangle.

    Rows run horizontally (along y), stepping down in z, alternating
    direction each row like mowing a lawn. Returns a list of (y, z).
    """
    waypoints = []
    z = zone['z_max']                       # start at the top
    row = 0
    while z >= zone['z_min'] - 1e-9:
        # build one row of y positions
        n_pts = max(2, int(round(
            (zone['y_max'] - zone['y_min']) / point_spacing)) + 1)
        ys = [zone['y_min'] + i * (zone['y_max'] - zone['y_min']) / (n_pts - 1)
              for i in range(n_pts)]
        if row % 2 == 1:
            ys = ys[::-1]                   # alternate direction -> S shape
        for y in ys:
            waypoints.append((y, z))
        z -= line_spacing
        row += 1
    return waypoints


def preview(zone_name: str, waypoints, width: int = 46, height: int = 14):
    """ASCII preview: numbers show visit order (mod 10)."""
    zone = ZONES[zone_name]
    grid = [[' '] * width for _ in range(height)]
    for i, (y, z) in enumerate(waypoints):
        col = int((y - zone['y_min']) / (zone['y_max'] - zone['y_min'])
                  * (width - 1))
        rowi = int((zone['z_max'] - z) / (zone['z_max'] - zone['z_min'])
                   * (height - 1))
        grid[rowi][col] = str(i % 10)
    lines = ['+' + '-' * width + '+']
    lines += ['|' + ''.join(r) + '|' for r in grid]
    lines += ['+' + '-' * width + '+']
    return '\n'.join(lines)


class RasterPlannerNode(Node):
    """Standalone preview runner (B3 will import ZONES/generate_raster)."""

    def __init__(self):
        super().__init__('raster_planner')
        self.declare_parameter('zone', 'forehead')
        self.declare_parameter('line_spacing', 0.01)
        self.declare_parameter('point_spacing', 0.01)

        zone_name = self.get_parameter('zone').value
        if zone_name not in ZONES:
            self.get_logger().error(
                f"Unknown zone '{zone_name}'. Options: {list(ZONES)}")
            return

        ls = self.get_parameter('line_spacing').value
        ps = self.get_parameter('point_spacing').value
        wps = generate_raster(ZONES[zone_name], ls, ps)

        self.get_logger().info(
            f"Zone '{zone_name}': {len(wps)} waypoints, "
            f"{ls*100:.1f}cm row spacing, {ps*100:.1f}cm point spacing")
        first = ', '.join(f'({y:.3f}, {z:.3f})' for y, z in wps[:4])
        self.get_logger().info(f'First waypoints: {first} ...')
        print(preview(zone_name, wps))


def main(args=None):
    rclpy.init(args=args)
    node = RasterPlannerNode()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
