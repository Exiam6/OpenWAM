"""Failure-path and resource-admission regression checks, with no GPU work."""
import copy,datetime as dt,json,pathlib,tempfile,unittest
from unittest.mock import patch
import bot
NOW=dt.datetime(2026,10,21,8,0,tzinfo=dt.timezone.utc) # 03:00 CDT

def sample():
 return dict(time=NOW.isoformat(),inventory_age_seconds=0,controller={'ActiveState':'active'},snapshot={'hosts':{}},jobs=[],experiments={},read_errors=[])
def config(state):
 return dict(state_dir=state,stall_seconds=1800,recoveries={},job_experiments={},agent_min_interval_seconds=850)
def job(jid='ex8-test',claim=None):
 return dict(id=jid,scientific_identity='ex8-test/scope',experiment='ex8-test',max_seconds=3600,hosts=['gpu01'],claim=claim,created_timestamp=NOW.timestamp()-300,paused_by=[],authorization_valid=True,last_log_progress_timestamp=NOW.timestamp())
def card(index,host='gpu01',when=NOW):
 return {host:{str(index):dict(index=index,qualified=True,idle=True,processes=[],utilization=0,used_mib=30,last=when.timestamp())}}

class AssessmentTests(unittest.TestCase):
 def test_original_overnight_failure_cannot_be_healthy(self):
  d=sample();d['jobs']=[job(claim={'status':'failed','worker_result':{'error':'outer supervisor assertion'}})];a=bot.assess(d,config('/nonexistent'),NOW)
  self.assertEqual(a['status'],'needs_action');self.assertEqual({x['code'] for x in a['issues']},{'failed_experiment','empty_queue_before_reproduction'})
  self.assertTrue(all(i['needs_agent'] for i in a['issues']))
 def test_recovery_queue_addresses_old_failure(self):
  d=sample();d['jobs']=[job('old',{'status':'failed'}),job('recovery')];c=config('/nonexistent');c['recoveries']={'old':'recovery'};a=bot.assess(d,c,NOW)
  self.assertEqual(a['status'],'waiting_for_gpu');self.assertFalse(a['issues']);self.assertEqual(len(a['addressed_failures']),1)
 def test_idle_eligible_gpu_plus_pending_job_is_actionable(self):
  d=sample();d['jobs']=[job()];d['snapshot']['hosts']=card(1);a=bot.assess(d,config('/nonexistent'),NOW)
  self.assertIn('eligible_gpu_not_dispatched',[x['code'] for x in a['issues']])
 def test_no_false_dispatch_alarm_during_controller_grace(self):
  d=sample();j=job();j['created_timestamp']=NOW.timestamp()-10;d['jobs']=[j];d['snapshot']['hosts']=card(1)
  self.assertEqual(bot.assess(d,config('/nonexistent'),NOW)['status'],'waiting_for_gpu')
 def test_priority_card_allowed_after_window_but_others_not(self):
  after=dt.datetime(2026,10,21,15,0,tzinfo=dt.timezone.utc);j=job();d={'hosts':card(4,when=after)}
  self.assertFalse(bot.eligible_cards(j,d,after));d['hosts']=card(1,when=after);self.assertEqual(bot.eligible_cards(j,d,after),['gpu01/1'])
 def test_window_must_fit_full_job_and_stale_idle_never_eligible(self):
  near=dt.datetime(2026,10,21,14,20,tzinfo=dt.timezone.utc);d={'hosts':card(4,when=near)};self.assertFalse(bot.eligible_cards(job(),d,near))
  d={'hosts':card(1,when=NOW-dt.timedelta(seconds=26))};self.assertFalse(bot.eligible_cards(job(),d,NOW))
 def test_no_takeover_of_foreign_gpu(self):
  d={'hosts':card(1)};d['hosts']['gpu01']['1']['processes']=[{'user':'other'}];self.assertFalse(bot.eligible_cards(job(),d,NOW))
 def test_long_silent_job_is_reported_even_when_another_job_progresses(self):
  d=sample();j=job(claim={'status':'running','created':(NOW-dt.timedelta(hours=1)).isoformat()});j['last_log_progress_timestamp']=NOW.timestamp()-1900;d['jobs']=[j];d['experiments']={'ex8-test':{'log_progress_timestamp':NOW.timestamp(),'files':{}}}
  self.assertIn('experiment_no_log_progress',[x['code'] for x in bot.assess(d,config('/nonexistent'),NOW)['issues']])
 def test_stale_controller_needs_recovery(self):
  d=sample();d['inventory_age_seconds']=180;d['controller']['ActiveState']='failed';a=bot.assess(d,config('/nonexistent'),NOW)
  self.assertTrue({'controller_down','inventory_stale'}<={i['code'] for i in a['issues']})
 def test_readiness_does_not_count_as_paper_reproduction(self):
  d=sample();d['experiments']={'recovery':{'files':{'readout.json':{'status':'readiness_complete_full_not_started','next_gate':'full_cost_review'}}}};a=bot.assess(d,config('/nonexistent'),NOW)
  self.assertFalse(a['full_endpoint_complete']);self.assertFalse(a['full_paper_reproduced']);self.assertIn('readiness_needs_full_stage',[x['code'] for x in a['issues']])
 def test_global_pause_prevents_all_mutations(self):
  with tempfile.TemporaryDirectory() as td:
   state=pathlib.Path(td);(state/'paused.json').write_text('{}');c=config(td)
   with patch.object(bot,'collect',return_value=sample()),patch.object(bot,'repair_controller') as repair,patch.object(bot,'run_agent') as agent:
    a=bot.tick(c,state,force_agent=True)
   self.assertEqual(a['assessment']['status'],'paused');repair.assert_not_called();agent.assert_not_called()
 def test_failed_takeover_does_not_disable_future_ticks(self):
  with tempfile.TemporaryDirectory() as td:
   state=pathlib.Path(td);c=config(td);d=sample()
   with patch.object(bot,'collect',return_value=d),patch.object(bot,'repair_controller',return_value=[]),patch.object(bot,'run_agent',return_value={'status':'failed'}) as agent:
    bot.tick(c,state);bot.tick(c,state)
   self.assertEqual(agent.call_count,2);self.assertEqual(len((state/'history.jsonl').read_text().splitlines()),2)
 def test_unchanged_required_user_decision_does_not_spin_agent(self):
  with tempfile.TemporaryDirectory() as td:
   state=pathlib.Path(td);c=config(td);d=sample();a=bot.assess(d,c,NOW)
   bot.write(state/'agent.json',{'started':(NOW-dt.timedelta(hours=1)).isoformat(),'response':{'status':'needs_user','decision_basis':'explicit_user_constraint','decision_evidence':['User imposed an explicit outside constraint']},'issue_fingerprint':bot.issue_fingerprint(a)})
   with patch.object(bot,'collect',return_value=d),patch.object(bot,'repair_controller',return_value=[]),patch.object(bot,'run_agent') as agent:result=bot.tick(c,state)
   agent.assert_not_called();self.assertEqual(result['agent'],'awaiting_required_user_input_checks_continue')
 def test_incomplete_inventory_is_reported(self):
  d=sample();d['expected_gpu_count']=16;d['snapshot']['errors']={'gpu02':'SSH unavailable'};a=bot.assess(d,config('/nonexistent'),NOW)
  self.assertTrue({'inventory_errors','incomplete_gpu_inventory'}<={i['code'] for i in a['issues']})
 def test_controller_repair_only_starts_our_controller(self):
  d=sample();d['controller']['ActiveState']='failed';a=bot.assess(d,config('/nonexistent'),NOW)
  with tempfile.TemporaryDirectory() as td,patch.object(bot,'command',return_value={'returncode':0,'stdout':'','stderr':''}) as call:
   actions=bot.repair_controller(d,a,pathlib.Path(td))
  self.assertEqual(actions[0]['action'],'start');self.assertEqual(call.call_args.args[0],['systemctl','--user','start','openwam-nightly-gpu.service'])
 def test_busy_takeover_still_observes_next_timer_without_duplicate(self):
  import fcntl
  with tempfile.TemporaryDirectory() as td:
   root=pathlib.Path(td);c=config(td);bot.write(root/'config.json',c)
   with (root/'bot.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    with patch('sys.argv',['bot','--config',str(root/'config.json')]),patch.object(bot,'collect',return_value=sample()),patch.object(bot,'run_agent') as agent:bot.main()
   agent.assert_not_called();self.assertTrue(bot.read(root/'latest.json')['concurrent_takeover_observation']);self.assertTrue((root/'heartbeat.json').is_file())
 def test_diagnostic_timeout_or_abandoned_agent_is_actionable_even_with_queue(self):
  for takeover in [{'status':'timed_out'},{'status':'running','started':(NOW-dt.timedelta(minutes=20)).isoformat()}]:
   d=sample();d['jobs']=[job()];d['last_takeover']=takeover;a=bot.assess(d,config('/nonexistent'),NOW)
   self.assertEqual(a['status'],'needs_action');self.assertTrue(any(i['code'].startswith('diagnostic_takeover_') for i in a['issues']))
 def test_agent_planning_estimate_or_unsubstantiated_hold_does_not_suppress_work(self):
  for response in [{'status':'needs_user'},{'status':'needs_user','decision_basis':'none','decision_evidence':['Agent initial planning estimate']}]:
   self.assertFalse(bot.valid_user_hold({'response':response}))
  self.assertTrue(bot.valid_user_hold({'response':{'status':'needs_user','decision_basis':'automatic_approval_rejection','decision_evidence':['Recorded real review rejection']}}))
 def test_registered_full_queue_does_not_retrigger_old_readiness_gate(self):
  d=sample();j=job();j['stage']='native_full';d['jobs']=[j];d['experiments']={'pilot':{'files':{'readout.json':{'status':'readiness_complete_full_not_started'}}},'full':{'files':{'protocol.json':{'full_registered_for_dispatch':True},'readout.json':{'total_task_mode_rows':100,'full_endpoint_complete':False}}}}
  a=bot.assess(d,config('/nonexistent'),NOW);self.assertFalse(a['issues']);self.assertEqual(a['status'],'waiting_for_gpu')
 def test_scaling_hold_is_actionable_while_first_row_is_running(self):
  d=sample();j=job(claim={'status':'running','created':NOW.isoformat()});d['jobs']=[j];d['experiments']={'full':{'files':{'protocol.json':{'full_registered_for_dispatch':True},'stage-state.json':{'status':'gate_failed','error':'native first row failed'}}}}
  self.assertIn('full_scaling_gate_hold',[i['code'] for i in bot.assess(d,config('/nonexistent'),NOW)['issues']])
 def test_invalid_authorization_does_not_get_silently_admitted(self):
  d=sample();j=job();j['authorization_valid']=False;d['jobs']=[j];a=bot.assess(d,config('/nonexistent'),NOW)
  self.assertIn('invalid_job_manifest',[i['code'] for i in a['issues']])


class ProcessTests(unittest.TestCase):
 def setup_fake(self,td,body):
  root=pathlib.Path(td);exe=root/'fake-codex';exe.write_text('#!/usr/bin/python3\n'+body);exe.chmod(0o700)
  c=config(td);c.update(codex_binary=str(exe),worktree=td,agent_timeout_seconds=3)
  return root,c
 def test_real_subprocess_takeover_on_empty_queue_preserves_failure(self):
  body="import sys,json\np=sys.argv[sys.argv.index('-o')+1]\nprompt=sys.stdin.read()\nassert 'official' in prompt\njson.dump(dict(status='failed',summary='diagnosis remains incomplete',actions=[],evidence=[],next_step='inspect'),open(p,'w'))\nprint(json.dumps({'type':'turn.completed'}))\n"
  with tempfile.TemporaryDirectory() as td:
   root,c=self.setup_fake(td,body)
   with patch.object(bot,'collect',return_value=sample()),patch.object(bot,'repair_controller',return_value=[]):result=bot.tick(c,root)
   self.assertEqual(result['agent']['status'],'complete');self.assertEqual(result['agent']['response']['status'],'failed');self.assertEqual(result['assessment']['status'],'needs_action');self.assertTrue((root/'agent.json').exists())
   self.assertFalse(result['assessment']['full_endpoint_complete'])
 def test_agent_timeout_is_bounded_and_recorded(self):
  with tempfile.TemporaryDirectory() as td:
   root,c=self.setup_fake(td,'import time\ntime.sleep(30)\n');c['agent_timeout_seconds']=0.1
   result=bot.run_agent(c,root,{'issues':[]},bot.utcnow())
   self.assertEqual(result['status'],'timed_out');self.assertLess(result['seconds'],5)
   self.assertEqual(bot.read(root/'agent.json')['status'],'timed_out');self.assertIn('agent_finished',(root/'alerts.jsonl').read_text())

if __name__=='__main__':unittest.main()
