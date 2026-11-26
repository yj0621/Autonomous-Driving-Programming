import rclpy
from rcl_interfaces.msg import ParameterType, ParameterDescriptor, FloatingPointRange
from rclpy.node import Node
from f110_msgs.msg import WpntArray


class SpeedScaler(Node):
    """Scales the velocity of the global waypoints by a single global factor."""

    def __init__(self):
        super().__init__('speed_scaler',
                         allow_undeclared_parameters=True,
                         automatically_declare_parameters_from_overrides=True)

        if not self.has_parameter('global_limit'):
            self.declare_parameter('global_limit', 0.5)
        self.set_descriptor('global_limit', descriptor=ParameterDescriptor(
            type=ParameterType.PARAMETER_DOUBLE,
            floating_point_range=[FloatingPointRange(from_value=0.0, to_value=1.0, step=0.01)]))

        self.glb_wpnts_og = None
        self.glb_wpnts_sp_og = None

        self.create_subscription(WpntArray, '/global_waypoints', self.global_waypoints_cb, 10)
        self.create_subscription(WpntArray, '/global_waypoints/shortest_path',
                                 self.global_waypoints_sp_cb, 10)

        self.scaled_points_pub = self.create_publisher(WpntArray, '/global_waypoints_scaled', 10)
        self.scaled_points_sp_pub = self.create_publisher(
            WpntArray, '/global_waypoints_scaled/shortest_path', 10)

        self.create_timer(0.5, self.timer_callback)
        self.get_logger().info(
            f'Waiting for global waypoints, global_limit={self.global_limit:.2f}')

    @property
    def global_limit(self) -> float:
        return float(min(max(self.get_parameter('global_limit').value, 0.0), 1.0))

    def global_waypoints_cb(self, data: WpntArray):
        self.glb_wpnts_og = data

    def global_waypoints_sp_cb(self, data: WpntArray):
        self.glb_wpnts_sp_og = data

    def scale(self, wpnts: WpntArray, limit: float) -> WpntArray:
        scaled = WpntArray()
        scaled.header = wpnts.header
        scaled.wpnts = list(wpnts.wpnts)
        for wpnt in scaled.wpnts:
            wpnt.vx_mps = wpnt.vx_mps * limit
        return scaled

    def timer_callback(self):
        limit = self.global_limit
        if self.glb_wpnts_og is not None:
            self.scaled_points_pub.publish(self.scale(self.glb_wpnts_og, limit))
        if self.glb_wpnts_sp_og is not None:
            self.scaled_points_sp_pub.publish(self.scale(self.glb_wpnts_sp_og, limit))


def main():
    rclpy.init()
    rclpy.spin(SpeedScaler())
    rclpy.shutdown()
