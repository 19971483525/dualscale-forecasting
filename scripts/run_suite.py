from pathlib import Path
import argparse
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from hubei_revision.suite import run_groups

p=argparse.ArgumentParser(description='Run the fixed aligned-history revision experiments.')
p.add_argument('--groups',default='all',help='Comma-separated E01..E11, or all.')
a=p.parse_args()
groups=['E01','E02','E03','E04','E05','E06','E08','E07','E09','E10','E11'] if a.groups=='all' else a.groups.split(',')
run_groups(groups)
