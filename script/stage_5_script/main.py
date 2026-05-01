import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from local_code.path_utils import resolve_stage_data_dir


def main():
    data_dir = resolve_stage_data_dir("stage_5_data", ("cora", "citeseer", "pubmed"))
    print(f"using stage 5 data directory: {data_dir}")
    for dataset_name in ("cora", "citeseer", "pubmed"):
        dataset_dir = data_dir / dataset_name
        print(f"{dataset_name}:")
        print(f"  node file: {dataset_dir / 'node'}")
        print(f"  link file: {dataset_dir / 'link'}")
    print("Stage 5 GCN training code is not implemented yet.")


if __name__ == "__main__":
    main()
