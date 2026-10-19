import datetime as dt
import unittest

from policy import advance,eligible,window


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.now=dt.datetime.fromisoformat('2026-10-19T06:00:00+00:00')
        self.gpu=dict(index=3,uuid='GPU-test',utilization=0,used_mib=32,processes=[],idle=True)
    def tick(self,prior,seconds,**changes):
        return advance(prior,dict(gpus=[dict(self.gpu,**changes)]),self.now+dt.timedelta(seconds=seconds),'policy')
    def qualified(self):
        rows={}
        for second in range(0,1801,10):rows=self.tick(rows,second)
        return rows
    def test_full_thirty_minutes_required(self):
        rows={}
        for second in range(0,1800,10):rows=self.tick(rows,second)
        self.assertFalse(rows['3']['qualified'])
        rows=self.tick(rows,1800);self.assertTrue(rows['3']['qualified'])
        self.assertTrue(eligible(rows['3'],self.now+dt.timedelta(seconds=1800),600))
    def test_zero_utilization_with_process_is_busy(self):
        rows=self.tick(self.qualified(),1810,processes=[dict(pid=1)],idle=False)
        self.assertFalse(rows['3']['qualified']);self.assertIsNone(rows['3']['since'])
    def test_busy_or_memory_resets_history(self):
        for changes in [dict(utilization=1),dict(used_mib=256),dict(idle=False)]:
            rows=self.tick(self.qualified(),1810,**changes)
            self.assertFalse(rows['3']['qualified'])
    def test_observation_gap_or_uuid_change_resets(self):
        self.assertFalse(self.tick(self.qualified(),1830)['3']['qualified'])
        self.assertFalse(self.tick(self.qualified(),1810,uuid='different')['3']['qualified'])
    def test_policy_change_or_backwards_clock_resets(self):
        rows=advance(self.qualified(),dict(gpus=[self.gpu]),self.now+dt.timedelta(seconds=1810),'changed')
        self.assertFalse(rows['3']['qualified'])
        self.assertFalse(self.tick(self.qualified(),1790)['3']['qualified'])
    def test_missing_host_cannot_carry_history(self):
        self.assertEqual(advance(self.qualified(),dict(gpus=[]),self.now,'policy'),{})
    def test_stale_qualification_rejected(self):
        self.assertFalse(eligible(self.qualified()['3'],self.now+dt.timedelta(seconds=1830),600))
    def test_window_boundaries(self):
        for timestamp,expected in [('2026-10-19T05:59:59+00:00',False),('2026-10-19T06:00:00+00:00',True),('2026-10-19T14:29:59+00:00',True),('2026-10-19T14:30:00+00:00',False)]:
            self.assertEqual(window(dt.datetime.fromisoformat(timestamp))[2],expected)
    def test_dst_uses_chicago_clock(self):
        for timestamp in ['2026-11-01T06:30:00+00:00','2026-11-01T07:30:00+00:00']:
            _,end,inside=window(dt.datetime.fromisoformat(timestamp))
            self.assertTrue(inside);self.assertEqual(end.hour,15);self.assertEqual(end.minute,30)
        self.assertFalse(window(dt.datetime.fromisoformat('2026-11-02T06:59:59+00:00'))[2])
    def test_full_job_and_cleanup_must_fit(self):
        now=dt.datetime.fromisoformat('2026-10-19T14:00:00+00:00')
        row=dict(self.qualified()['3'],last=now.timestamp())
        self.assertTrue(eligible(row,now,1679))
        self.assertFalse(eligible(row,now,1680))


if __name__=='__main__':unittest.main()
