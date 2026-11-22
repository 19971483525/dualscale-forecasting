from pathlib import Path
import argparse
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from hubei_revision.analyses import analyze

p=argparse.ArgumentParser();p.add_argument('--groups',default='E03');a=p.parse_args()
for group in a.groups.split(','):analyze(group)
