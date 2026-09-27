"""Two fixed actions through the native client/converter and simulator path."""
import json,os
from pathlib import Path
import norm_client
from attribution_trace import AttributionTrace,FixedCommandTransport
from capture_repeat_inputs import make_recording_model
from trajectory_trace import instrument_interface
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/attribution-20260921'

def main():
    original_main=norm_client.client.main
    def patched_main():
        import openwam2robotwin_interface as interface
        capture=json.loads((ROOT/'results/repeatability-20260921/capture-summary.json').read_text())
        hashes={(r['condition'],r['seed']):r['payload_sha256']for r in capture['records']if r['condition']=='clean'}
        commands=json.loads((OUT/'fixed-commands.json').read_text())['commands']
        trace=AttributionTrace(os.environ['ROBUST_RUN_DIR'],hashes,interface._extract_eef_proprio)
        assert 'pick_dual_bottles'not in interface._STEP_LIM_OVERRIDES
        interface._STEP_LIM_OVERRIDES={**interface._STEP_LIM_OVERRIDES,'pick_dual_bottles':2}
        def model(args):
            m=make_recording_model(interface);m._client=FixedCommandTransport(trace,commands);return m
        interface.get_model=model
        base_eval=interface.eval
        def eval_with_env(env,m,observation):
            trace.env=env
            return base_eval(env,m,observation)
        interface.eval=eval_with_env
        change=norm_client.client.change_observation
        def traced_change(observation,condition,seed,step):
            assert condition=='clean'
            altered=change(observation,condition,seed,step)
            trace.begin(observation,altered,condition,seed,step)
            return altered
        norm_client.client.change_observation=traced_change
        instrument_interface(interface,trace)
        original_main();assert trace.current is None
        path=Path(os.environ['ROBUST_RUN_DIR'])/'summary.json';s=json.loads(path.read_text())
        assert len(s['records'])==3 and all(r['actions']==r['calls']==2 for r in s['records'])
        s['role']='two-action open-loop diagnostic; native success/failure flags are not policy evaluation outcomes'
        s['policy_inference_calls']=0;s['step_limit_override']=2;s['original_upstream_step_limit']=400
        path.write_text(json.dumps(s,indent=2)+'\n')
    norm_client.client.main=patched_main
    norm_client.main()
if __name__=='__main__':main()
