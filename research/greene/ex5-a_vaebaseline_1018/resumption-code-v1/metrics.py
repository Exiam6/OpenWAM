"""Preregistered matched Raw-DINO minus Wan-VAE paired comparison."""
import numpy as np


def compare(left,right):
    left=np.asarray(left,dtype=float);right=np.asarray(right,dtype=float)
    assert left.shape==right.shape==(3,3,20)
    assert np.isin(left,[0,1]).all() and np.isin(right,[0,1]).all()
    difference=left-right;rng=np.random.default_rng(921018);draws=10000
    seeds=rng.integers(0,3,size=(draws,3));scenes=rng.integers(0,20,size=(draws,3,20))
    samples=difference[seeds[:,:,None,None],np.arange(3)[None,None,:,None],scenes[:,None,:,:]]
    low,high=np.quantile(samples.mean(axis=(1,2,3)),[.025,.975])
    per_seed=difference.mean(axis=(1,2));point=float(difference.mean())
    supported=point>=.05-1e-12 and (per_seed>0).sum()>=2 and low>0
    verdict='useful_raw_dino_benefit_supported' if supported else ('positive_benefit_refuted_on_this_cohort' if high<=0 else 'undecided')
    return dict(difference=point,ci95=[float(low),float(high)],per_seed_difference=per_seed.tolist(),
        per_task_difference=difference.mean(axis=(0,2)).tolist(),paired_gains=int((difference>0).sum()),paired_losses=int((difference<0).sum()),
        verdict=verdict,draws=draws,bootstrap_seed=921018,
        limitation='Three training seeds and known development scenes; equal steps/data do not imply equal FLOPs. Encoder pretraining and latent representation are not independently identified.')
