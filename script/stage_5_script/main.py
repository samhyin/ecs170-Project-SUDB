import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from script_gcn_node_classification import run_stage5


def main():
    run_stage5()


if __name__ == "__main__":
    main()
