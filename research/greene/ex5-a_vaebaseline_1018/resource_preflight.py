"""Run the original preflight with the separately registered CPU resource probe."""
import argparse
import sys
from pathlib import Path

from resource_probe import ROOT,verify,idle_gpus

sys.path.insert(0,str(ROOT/'code'))
import run_preflight as frozen


def main(gpu,check=False):
    amendment=verify();assert str(Path(__file__).resolve()) in amendment['files']
    # Only the CPU eligibility query changes; all original gates and stage commands remain.
    frozen.idle_gpus=idle_gpus
    if check:frozen.admission();print('admitted; numerical stages unchanged')
    else:frozen.main(gpu)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--gpu',type=int,choices=[0,5,6,7],required=True);parser.add_argument('--check-admission',action='store_true');args=parser.parse_args();main(args.gpu,args.check_admission)
