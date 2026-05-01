import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from local_code.path_utils import resolve_stage_data_dir


def main():
    data_dir = resolve_stage_data_dir(
        "stage_4_data",
        ("text_classification", "text_generation"),
    )
    print(f"using stage 4 data directory: {data_dir}")
    print("text classification data:", data_dir / "text_classification")
    print("text generation data:", data_dir / "text_generation")
    print("Stage 4 RNN training code is not implemented yet.")


if __name__ == "__main__":
    main()
