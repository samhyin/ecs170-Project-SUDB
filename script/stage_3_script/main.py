import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from local_code.path_utils import ensure_result_dir, resolve_stage_data_dir
from local_code.stage_3_code.Dataset_Loader_Image_Classification import (
    Dataset_Loader_Image_Classification,
)
from local_code.stage_3_code.Evaluate_Classification import Evaluate_Classification
from local_code.stage_3_code.Method_CNN import Method_CNN
from local_code.stage_3_code.Result_Saver import Result_Saver
from local_code.stage_3_code.Setting_Image_Classification import (
    Setting_Image_Classification,
)


DATASET_CONFIGS = {
    "MNIST": {
        "description": "handwritten digit images",
        "image_mode": "grayscale",
        "batch_size": 128,
        "epochs": 4,
        "quick_epochs": 1,
    },
    "ORL": {
        "description": "human face images",
        "image_mode": "grayscale",
        "batch_size": 32,
        "epochs": 30,
        "quick_epochs": 3,
    },
    "CIFAR": {
        "description": "colored object images",
        "image_mode": "rgb",
        "batch_size": 128,
        "epochs": 6,
        "quick_epochs": 1,
    },
}


EXPERIMENT_CONFIGS = {
    "baseline": {
        "description": "three convolution blocks with 32, 64, and 128 channels",
        "conv_channels": (32, 64, 128),
        "hidden_dim": 128,
        "dropout": 0.25,
        "kernel_size": 3,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
    },
    "shallow": {
        "description": "ablation with two smaller convolution blocks",
        "conv_channels": (16, 32),
        "hidden_dim": 64,
        "dropout": 0.10,
        "kernel_size": 3,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
    },
}


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def run_experiment(dataset_name, experiment_name, data_dir, result_dir, quick=False):
    dataset_config = DATASET_CONFIGS[dataset_name]
    experiment_config = EXPERIMENT_CONFIGS[experiment_name]
    epochs = dataset_config["quick_epochs"] if quick else dataset_config["epochs"]

    data_obj = Dataset_Loader_Image_Classification(
        dataset_name,
        dataset_config["description"],
    )
    data_obj.dataset_source_folder_path = str(data_dir)
    data_obj.dataset_source_file_name = dataset_name
    data_obj.image_mode = dataset_config["image_mode"]

    method_obj = Method_CNN(
        "convolutional neural network",
        experiment_config["description"],
        conv_channels=experiment_config["conv_channels"],
        hidden_dim=experiment_config["hidden_dim"],
        dropout=experiment_config["dropout"],
        kernel_size=experiment_config["kernel_size"],
        max_epoch=epochs,
        learning_rate=experiment_config["learning_rate"],
        batch_size=dataset_config["batch_size"],
        weight_decay=experiment_config["weight_decay"],
    )
    method_obj.curve_output_path = (
        result_dir / f"{dataset_name.lower()}_{experiment_name}_loss_curve.png"
    )

    result_obj = Result_Saver("saver", "")
    result_obj.result_destination_folder_path = str(
        result_dir / f"{dataset_name}_{experiment_name}_CNN_"
    )
    result_obj.result_destination_file_name = "prediction_result"

    setting_obj = Setting_Image_Classification("pre-partitioned train/test", "")
    evaluate_obj = Evaluate_Classification(
        "accuracy precision recall f1",
        "macro and weighted multiclass metrics",
    )

    print("************ Start ************")
    print(f"dataset: {dataset_name}")
    print(f"experiment: {experiment_name}")
    print(f"using stage 3 data directory: {data_dir}")
    print(f"using stage 3 result directory: {result_dir}")
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    metrics, _ = setting_obj.load_run_save_evaluate()
    print("************ Finish ************")

    return {
        "dataset": dataset_name,
        "experiment": experiment_name,
        "description": experiment_config["description"],
        "metrics": metrics,
        "curve_output_path": str(method_obj.curve_output_path),
        "prediction_output_path": (
            result_obj.result_destination_folder_path
            + result_obj.result_destination_file_name
            + "_"
            + str(result_obj.fold_count)
        ),
        "epochs": epochs,
        "batch_size": dataset_config["batch_size"],
        "model_config": {
            key: value
            for key, value in experiment_config.items()
            if key != "description"
        },
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Run Stage 3 CNN experiments.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASET_CONFIGS.keys()),
        choices=list(DATASET_CONFIGS.keys()),
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=list(EXPERIMENT_CONFIGS.keys()),
        choices=list(EXPERIMENT_CONFIGS.keys()),
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a short smoke test with fewer epochs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(2)
    torch.manual_seed(2)

    data_dir = resolve_stage_data_dir("stage_3_data", ("CIFAR", "MNIST", "ORL"))
    result_dir = ensure_result_dir("stage_3_result")

    summaries = []
    for dataset_name in args.datasets:
        for experiment_name in args.experiments:
            summaries.append(
                run_experiment(
                    dataset_name,
                    experiment_name,
                    data_dir,
                    result_dir,
                    quick=args.quick,
                )
            )

    summary_path = result_dir / "stage_3_summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(json_safe(summaries), handle, indent=2)

    print("************ Stage 3 Summary ************")
    for summary in summaries:
        metrics = summary["metrics"]
        print(
            f"{summary['dataset']} {summary['experiment']}: "
            f"acc={metrics['accuracy']:.4f}, "
            f"precision={metrics['precision_macro']:.4f}, "
            f"recall={metrics['recall_macro']:.4f}, "
            f"f1={metrics['f1_macro']:.4f}"
        )
    print(f"saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
