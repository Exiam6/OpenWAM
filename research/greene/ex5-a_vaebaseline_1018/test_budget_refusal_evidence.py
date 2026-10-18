import json
from pathlib import Path
import tempfile
import unittest

from build_cache import sha
from report_complete import budget_refusal_evidence


class BudgetRefusalArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.proposal=dict(status='proposal_only_pending_user_decision',
            gpu_launch_authorized_by_proposal=False,affected_seeds=[42,43],
            total_gpu_cap_seconds=73800,seed44_continues_original_limit_seconds=10800,
            evidence_sha256={})
        for seed in [42,43]:
            cost=self.put(f'pilot-cost-observation-seed{seed}.json',dict(seed=seed,
                kind='cost_observation_only_not_a_release',pilot_occupied_seconds=2400,
                remaining_seconds=8400,headroom_after_required60s=-80,
                projection=dict(projected_remaining_seconds=8420)))
            refusal=self.put(f'evaluation-readiness-seed{seed}-original.json',dict(seed=seed,
                exit_code=2,stdout='Remaining fixed endpoint does not fit per-seed evaluation budget\n',
                original_gate_preserved=True,full_evaluation_launched=False,release_exists=False,
                command=['python',str(self.root/'corrected-protocol-v1/release_evaluation.py'),
                         '--seed',str(seed),'--check-readiness'],budget_observation=str(cost)))
            for path in [cost,refusal]:self.proposal['evidence_sha256'][str(path)]=sha(path)
        self.save_proposal()

    def put(self,name,data):
        path=self.root/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(data));return path

    def save_proposal(self):self.put('evaluation-budget-amendment-proposal-v1.json',self.proposal)

    def modify(self,name,**changes):
        path=self.root/name;data=json.loads(path.read_text());data.update(changes)
        self.put(name,data);self.proposal['evidence_sha256'][str(path)]=sha(path);self.save_proposal()

    def test_preserves_all_original_evidence_without_creating_release(self):
        expected=set(self.root.iterdir())
        self.assertEqual(set(budget_refusal_evidence(self.root)),expected)
        self.assertEqual(set(self.root.iterdir()),expected)

    def test_missing_refusal_is_rejected(self):
        (self.root/'evaluation-readiness-seed42-original.json').unlink()
        with self.assertRaises(AssertionError):budget_refusal_evidence(self.root)

    def test_modified_cost_bytes_are_rejected(self):
        self.put('pilot-cost-observation-seed43.json',{})
        with self.assertRaises(AssertionError):budget_refusal_evidence(self.root)

    def test_refusal_cannot_be_relabelled_success_even_with_updated_hash(self):
        self.modify('evaluation-readiness-seed42-original.json',exit_code=0)
        with self.assertRaises(AssertionError):budget_refusal_evidence(self.root)

    def test_proposal_cannot_authorize_launch(self):
        self.proposal['gpu_launch_authorized_by_proposal']=True;self.save_proposal()
        with self.assertRaises(AssertionError):budget_refusal_evidence(self.root)

    def test_cost_cannot_be_relabelled_release(self):
        self.modify('pilot-cost-observation-seed42.json',kind='release')
        with self.assertRaises(AssertionError):budget_refusal_evidence(self.root)

    def test_refusal_and_cost_must_match_original_cap(self):
        self.modify('pilot-cost-observation-seed43.json',remaining_seconds=8520)
        with self.assertRaises(AssertionError):budget_refusal_evidence(self.root)

    def test_refusals_cannot_disappear_from_evidence_index(self):
        del self.proposal['evidence_sha256'][str(self.root/'evaluation-readiness-seed43-original.json')]
        self.save_proposal()
        with self.assertRaises(AssertionError):budget_refusal_evidence(self.root)

    def test_unaffected_legacy_archive_has_no_new_requirement(self):
        with tempfile.TemporaryDirectory() as root:self.assertEqual(budget_refusal_evidence(root),[])


if __name__=='__main__':unittest.main()
