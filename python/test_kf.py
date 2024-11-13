from kf import KF
import numpy as np
import unittest 
class testKF(unittest.TestCase):

    def test_can_const_with_x_and_v(self):
        x = 0.2
        v = 2.3
        kf = KF(x , v, acc_var= 1.2) 
        self.assertAlmostEqual(kf.pos , x)
        self.assertAlmostEqual(kf.vel , v)
    

    def test_aft_call_pred_mean_and_cov_are_of_rt_shape(self):
        x = 0.2
        v = 2.3
        kf = KF(x , v, acc_var= 1.2) 
        kf.predict(dt = 0.1)

        self.assertEqual(kf.cov.shape , (2,2))
        self.assertEqual(kf.mean.shape , (2,))

    
    def test_aft_call_pred_incr_state_uncertainty(self):
        x = 0.2
        v = 2.3
        kf = KF(x , v, acc_var= 1.2) 

        for i in range(10):
            det_before = np.linalg.det(kf.cov)
            kf.predict(dt = 0.1)
            det_after = np.linalg.det(kf.cov)

            self.assertGreater(det_after,det_before)
            print(det_before , det_after)

    
    def test_can_call_update_decreases_uncertainty(self):
        x = 0.2
        v = 2.3

        kf = KF(x , v, acc_var= 1.2) 

        det_before = np.linalg.det(kf.cov)
        kf.update(0.1 , 0.01)
        det_after = np.linalg.det(kf.cov)

        self.assertLess(det_after , det_before)
