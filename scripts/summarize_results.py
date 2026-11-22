"""Rebuild local aggregate tables after authorized experiments complete."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from hubei_revision.reporting import build_tables
if __name__=='__main__':
    build_tables()
    print('Local aggregate tables regenerated; use export_paper_results.py for the public subset.')
