"""Challenge paired inference with known effects and a scalar independent sampler."""
import unittest
import numpy as np
from metrics import compare


class MetricsTests(unittest.TestCase):
    def test_identical_is_not_positive_benefit(self):
        x=np.zeros((3,3,20));result=compare(x,x)
        self.assertEqual(result['ci95'],[0.,0.]);self.assertEqual(result['paired_gains'],0)
        self.assertEqual(result['verdict'],'positive_benefit_refuted_on_this_cohort')

    def test_full_positive_and_negative_effect(self):
        one=np.ones((3,3,20));zero=np.zeros_like(one)
        self.assertEqual(compare(one,zero)['verdict'],'useful_raw_dino_benefit_supported')
        self.assertEqual(compare(zero,one)['ci95'],[-1.,-1.])

    def test_independent_scalar_resampling(self):
        rng=np.random.default_rng(17);left=rng.integers(0,2,(3,3,20));right=rng.integers(0,2,(3,3,20))
        result=compare(left,right);diff=left-right
        rng=np.random.default_rng(921018);seeds=rng.integers(0,3,(10000,3));scenes=rng.integers(0,20,(10000,3,20))
        values=[]
        for i in range(10000):
            total=sum(diff[s,t,scenes[i,t]].sum() for s in seeds[i] for t in range(3))
            values.append(total/180)
        np.testing.assert_allclose(result['ci95'],np.quantile(values,[.025,.975]),rtol=0,atol=0)

    def test_missing_seed_rejected(self):
        with self.assertRaises(AssertionError):compare(np.zeros((2,3,20)),np.zeros((2,3,20)))


if __name__=='__main__':unittest.main()
