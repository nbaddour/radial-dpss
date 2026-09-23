import os
import sys

# Put the directory containing the task_12_* modules (code/) on sys.path so the tests can import them
# regardless of the working directory pytest is invoked from.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
