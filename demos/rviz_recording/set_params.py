import sys,json,time
import rclpy
from rclpy.node import Node
from rcl_interfaces.srv import SetParameters
from rcl_interfaces.msg import Parameter,ParameterValue
rclpy.init(); n=Node('recording_parameter_client')
client=n.create_client(SetParameters,sys.argv[1]+'/set_parameters')
if not client.wait_for_service(timeout_sec=20): raise RuntimeError('Parameter service unavailable '+sys.argv[1])
req=SetParameters.Request()
for key,v in json.loads(sys.argv[2]).items():
    pv=ParameterValue()
    if isinstance(v,bool):pv.type=1;pv.bool_value=v
    elif isinstance(v,str):pv.type=4;pv.string_value=v
    else:pv.type=3;pv.double_value=float(v)
    req.parameters.append(Parameter(name=key,value=pv))
f=client.call_async(req);rclpy.spin_until_future_complete(n,f,timeout_sec=20)
print(f.result())
if f.result() is None or not all(r.successful for r in f.result().results):raise RuntimeError('Parameter update failed')
n.destroy_node();rclpy.shutdown()
