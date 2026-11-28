# Runs the d2d command line from inside the d2d/ folder (where Claude Code works
# on the agents' tasks): python run.py <command> ...
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
runpy.run_module('d2d', run_name='__main__')
