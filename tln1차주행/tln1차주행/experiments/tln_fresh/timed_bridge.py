"""Serial simulator executor and measured physics time/steering instrumentation.
Physics, map and collision rules are inherited unchanged from the project bridge.
"""
import rclpy,time
from rclpy.executors import SingleThreadedExecutor
from std_msgs.msg import Float64MultiArray
from f1tenth_gym_ros.gym_bridge import GymBridge
class TimedBridge(GymBridge):
 def __init__(self):
  super().__init__();self.ego_accel_x=0.;self.ego_accel_y=0.;self.synchronous=bool(self.get_parameter('synchronous_control').value) if self.has_parameter('synchronous_control') else False;self.remaining_steps=0;self.physics_elapsed=0.;self.paced=bool(self.get_parameter("paced_control").value) if self.has_parameter("paced_control") else False;self.pacer_origin=None;self.physics_pub=self.create_publisher(Float64MultiArray,'/sim/physics_state',1);self.stamped_physics_pub=self.create_publisher(Float64MultiArray,'/sim/physics_state_stamped',1)
  if self.synchronous and self.paced:
   self.destroy_timer(self.drive_timer);self.drive_timer=self.create_timer(.001,self.drive_timer_callback)
 def drive_callback(self,msg):
  super().drive_callback(msg);self.remaining_steps=10
  if self.pacer_origin is None:self.pacer_origin=time.monotonic()-self.physics_elapsed
 def drive_timer_callback(self):
  if self.synchronous and self.remaining_steps<=0:return
  if self.synchronous and self.paced and (self.pacer_origin is None or self.physics_elapsed+.01>time.monotonic()-self.pacer_origin):return
  stepped=self.ego_drive_published and (not self.has_opp or self.opp_drive_published)
  super().drive_timer_callback()
  if stepped:self.physics_elapsed+=float(self.env.unwrapped.timestep);self.remaining_steps=max(0,self.remaining_steps-1)
  msg=Float64MultiArray();msg.data=[self.physics_elapsed,float(self.env.unwrapped.sim.agents[0].state[2])];self.physics_pub.publish(msg);stamped=Float64MultiArray();stamped.data=[self.physics_elapsed,float(self.env.unwrapped.sim.agents[0].state[2]),float(self.ts.sec),float(self.ts.nanosec)];self.stamped_physics_pub.publish(stamped)
rclpy.init();node=TimedBridge();executor=SingleThreadedExecutor();executor.add_node(node)
try:executor.spin()
except KeyboardInterrupt:pass
finally:
 executor.shutdown();node.destroy_node()
 if rclpy.ok():rclpy.shutdown()
