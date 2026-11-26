import unittest
import numpy as np
from safety import shield
class ShieldTests(unittest.TestCase):
 def test_clear_straight_preserves_speed(self):
  u,s=shield(np.full(1080,20.),[0,2]);self.assertAlmostEqual(u[1],2);self.assertFalse(s['limited'])
 def test_front_obstacle_inside_sim_ttc_envelope_stops(self):
  # LiDAR is 0.27m forward; 0.20m return -> obstacle 0.47m from base.
  scan=np.full(1080,20.);scan[530:550]=.20
  u,s=shield(scan,[0,2]);self.assertEqual(u[1],0);self.assertTrue(s['limited'])
 def test_near_wall_off_the_straight_swept_corridor_does_not_stop(self):
  scan=np.full(1080,20.);angles=np.linspace(-2.35,2.35,1080);scan[abs(angles-np.pi/2)<.03]=.8
  u,s=shield(scan,[0,1]);self.assertAlmostEqual(u[1],1)
if __name__=='__main__':unittest.main()
