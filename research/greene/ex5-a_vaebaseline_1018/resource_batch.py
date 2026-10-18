"""Run the original batch parent with the separately registered CPU resource probe."""
import argparse
import sys
from pathlib import Path

from resource_probe import ROOT,verify,capacity

sys.path.insert(0,str(ROOT/'code'))
import run_batch as frozen


def main(phase,seeds,gpus,check=False):
    amendment=verify();assert str(Path(__file__).resolve()) in amendment['files']
    frozen.capacity=capacity
    if check:frozen.admission(phase,seeds);print('admitted; numerical stages unchanged')
    else:frozen.main(phase,seeds,gpus)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['training','pilot_evaluation','full_evaluation'],required=True);parser.add_argument('--seeds',nargs='+',type=int,required=True);parser.add_argument('--gpus',nargs='+',type=int,choices=range(8));parser.add_argument('--check-admission',action='store_true');args=parser.parse_args()
    assert args.gpus or args.check_admission;main(args.phase,args.seeds,args.gpus,args.check_admission)
