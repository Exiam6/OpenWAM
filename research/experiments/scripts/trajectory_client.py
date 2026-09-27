"""Add passive tracing around the unchanged frozen norm-client replay."""
import json,os
from pathlib import Path
import norm_client
from trajectory_trace import TrajectoryTrace,instrument_interface
ROOT=Path(__file__).resolve().parents[1]

def main():
    original_main=norm_client.client.main
    def traced_main():
        import openwam2robotwin_interface as interface
        capture=json.loads((ROOT/'results/repeatability-20260921/capture-summary.json').read_text())
        hashes={(r['condition'],r['seed']):r['payload_sha256']for r in capture['records']}
        trace=TrajectoryTrace(Path(os.environ['ROBUST_RUN_DIR']),hashes,interface._extract_eef_proprio)
        change=norm_client.client.change_observation
        def traced_change(observation,condition,seed,step):
            altered=change(observation,condition,seed,step)
            trace.begin(observation,altered,condition,seed,step)
            return altered
        norm_client.client.change_observation=traced_change
        instrument_interface(interface,trace)
        original_main()
        assert trace.current is None
    norm_client.client.main=traced_main
    norm_client.main()
if __name__=='__main__':main()
