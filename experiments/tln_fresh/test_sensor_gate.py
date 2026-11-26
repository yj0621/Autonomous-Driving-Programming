import unittest
from sensor_gate import SensorGate
class SensorGateTests(unittest.TestCase):
 def test_normal_sensor_loss_stops(self):
  g=SensorGate()
  for t in [0.,.1,.2]:self.assertEqual(g.check(t,0.,0.)[0],'fresh')
  self.assertEqual(g.check(.31,0.,0.)[0],'stop_sensor_stale')
 def test_blackout_with_fresh_queued_sensor_recovers(self):
  g=SensorGate();g.check(1.,1.,1.)
  self.assertEqual(g.check(4.,1.,1.)[0],'drain_pending_callbacks')
  self.assertEqual(g.check(4.02,4.01,4.01)[0],'fresh')
 def test_grace_expires_without_sensor(self):
  g=SensorGate();g.check(1.,1.,1.);g.check(4.,1.,1.)
  self.assertEqual(g.check(4.06,1.,1.)[0],'stop_sensor_stale')
 def test_repeated_blackout_does_not_extend_grace(self):
  g=SensorGate();g.check(1.,1.,1.);g.check(4.,1.,1.)
  self.assertEqual(g.check(8.,1.,1.)[0],'stop_sensor_stale')
if __name__=='__main__':unittest.main()
