"""Meaningful input/output checks: shape, invalid ranges, history reset and physical units."""
import unittest
import numpy as np
from world import preprocess,History,World
class Contracts(unittest.TestCase):
 def test_sensor_invalid_and_hokuyo_padding(self):
  x=np.full(1080,5.);x[0]=np.nan;x[2]=np.inf;x[4]=-np.inf;x[-1]=3
  z=preprocess(x);self.assertEqual(z.shape,(541,));np.testing.assert_allclose(z[:3],[1,1,0]);self.assertAlmostEqual(float(z[-1]),.3,places=6)
 def test_history_order_and_reset(self):
  h=History();h.push(np.full(1080,2.));z=h.push(np.full(1080,4.));np.testing.assert_allclose(z[0],[.2,.2,.4]);z=History().push(np.full(1080,8.));np.testing.assert_allclose(z[0],[.8,.8,.8])
 def test_speed_feature_and_signed_steering_units(self):
  w=World();xy=w.xy(0);state=np.array([*xy,.1,1.25,w.tangent(0)]);f=w.route_features(state,previous_steer=-.18,s_hint=0)
  self.assertEqual(f.shape,(34,));self.assertAlmostEqual(float(f[-2]),.5);self.assertAlmostEqual(float(f[-1]),-.5)
 def test_temporal_model_can_command_below_old_floor(self):
  # Decoder's physical contract deliberately has no 3m/s floor and no EMA.
  v=np.clip(np.array([0,.2,1])*2.5,0,2.5);np.testing.assert_allclose(v,[0,.5,2.5])
if __name__=='__main__':unittest.main()
