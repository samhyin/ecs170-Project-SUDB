import pickle
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from local_code.path_utils import resolve_stage_data_dir


def summarize_dataset(dataset_name, dataset_path):
    with open(dataset_path, "rb") as handle:
        loaded = pickle.load(handle)

    train_count = len(loaded["train"])
    test_count = len(loaded["test"])
    sample = loaded["train"][0]
    image_shape = np.array(sample["image"]).shape
    sample_label = sample["label"]

    print(f"{dataset_name}:")
    print(f"  file: {dataset_path}")
    print(f"  train instances: {train_count}")
    print(f"  test instances: {test_count}")
    print(f"  sample image shape: {image_shape}")
    print(f"  sample label: {sample_label}")


def main():
    data_dir = resolve_stage_data_dir("stage_3_data", ("CIFAR", "MNIST", "ORL"))
    print(f"using stage 3 data directory: {data_dir}")
    for dataset_name in ("MNIST", "ORL", "CIFAR"):
        summarize_dataset(dataset_name, data_dir / dataset_name)


if __name__ == "__main__":
    main()
